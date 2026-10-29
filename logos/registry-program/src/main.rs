//! Neurolock proof-of-personhood registry on the Logos Execution Zone (LEZ).
//!
//! WHAT IS STORED (all public PDAs of this program; no biometric data ever):
//!   config            PDA("neurolock/config")              Config { admin, stations, party window }
//!   commitment C      PDA(H("nl/commit" ‖ C))              Enrolled { user, id_pub }   (write-once)
//!   user u            PDA(H("nl/user"   ‖ u))              C                           (write-once)
//!   delegation        PDA(H("nl/deleg"  ‖ nullifier))      Delegation { … }
//!   agent index       PDA(H("nl/agent"  ‖ agent ‖ app))    nullifier (so is_active(agent, app) is one read)
//!
//! WHO SIGNS (checked by the chain itself: an account passed as a signer must
//! have signed the transaction, and the program sees `is_authorized`):
//!   register       an approved enrolment station
//!   delegate/revoke the enrolled user's own LEZ account (its key is derived from the
//!                  identity secret S on the user's device)
//!   admin calls    the admin account recorded at `Init`
//!
//! TIME: `plan.block_window(a..b)` makes a transaction valid only in blocks [a, b).
//!   register  -> only inside the party window
//!   delegate  -> only at or after the claimed `now_block`, and before its own expiry
//!
//! ONE AGENT PER HUMAN PER APP: the program computes the nullifier itself,
//!   nullifier = H("neurolock/nullifier/v1" ‖ user ‖ app_id),
//! so it can't be chosen by the client, and a new delegation is refused while the
//! stored one for that nullifier is still active and unexpired.

use borsh::{BorshDeserialize, BorshSerialize};
use lee_core::{
    account::AccountId,
    program::{AccountMeta, BlockValidityWindow, PdaSeed, Plan, PlanInput, run_program},
};
use risc0_zkvm::sha::{Impl, Sha256 as _};

type Bytes32 = [u8; 32];
const MAX_APP: usize = 64;
const MAX_SCOPE: usize = 128;
const MAX_STATIONS: usize = 16;

// ───────────────────────────── instructions ─────────────────────────────

#[derive(BorshSerialize, BorshDeserialize)]
pub enum Instruction {
    /// accounts: [config, admin(signer)]
    Init { party_start: u64, party_end: u64 },
    /// accounts: [config, admin(signer)]
    AddStation { station: Bytes32 },
    /// accounts: [config, admin(signer)]
    SetPartyWindow { party_start: u64, party_end: u64 },
    /// accounts: [config, commitment slot, user slot, station(signer)]
    /// `party_*` must equal the stored window (checked in apply); the chain
    /// rejects the transaction outside [party_start, party_end).
    Register { commitment: Bytes32, user: Bytes32, id_pub: Bytes32, party_start: u64, party_end: u64 },
    /// accounts: [user slot, delegation slot, agent index slot, user(signer)]
    Delegate { app_id: String, agent: Bytes32, scope: String, expiry_block: u64, now_block: u64 },
    /// accounts: [delegation slot, user(signer)]
    Revoke { app_id: String },
}

// ───────────────────────────── stored records ─────────────────────────────

#[derive(BorshSerialize, BorshDeserialize, Clone, Debug, PartialEq)]
pub struct Config {
    pub admin: Bytes32,
    pub stations: Vec<Bytes32>,
    pub party_start: u64,
    pub party_end: u64,
}

#[derive(BorshSerialize, BorshDeserialize, Clone, Debug, PartialEq)]
pub struct Enrolled {
    pub user: Bytes32,
    pub id_pub: Bytes32,
}

#[derive(BorshSerialize, BorshDeserialize, Clone, Debug, PartialEq)]
pub struct Delegation {
    pub nullifier: Bytes32,
    pub user: Bytes32,
    pub agent: Bytes32,
    pub app_id: String,
    pub scope: String,
    pub expiry_block: u64,
    pub active: bool,
}

// ───────────────────────────── effects (one per account) ─────────────────────────────

#[derive(BorshSerialize, BorshDeserialize, Debug)]
pub enum Effect {
    /// config: must be empty, becomes this config
    InitConfig(Config),
    /// config: caller must be the admin; then change it
    AdminAddStation { admin: Bytes32, station: Bytes32 },
    AdminSetWindow { admin: Bytes32, party_start: u64, party_end: u64 },
    /// config: read-only guard — station is approved and the claimed window is the stored one
    CheckStation { station: Bytes32, party_start: u64, party_end: u64 },
    /// write-once slot
    WriteOnce(Vec<u8>),
    /// user slot: read-only guard — this user is enrolled
    RequireEnrolled,
    /// delegation slot: new delegation, refused while an active, unexpired one exists
    PutDelegation { record: Delegation, now_block: u64 },
    /// agent index slot: always overwritten with the latest nullifier for (agent, app)
    PutIndex(Bytes32),
    /// delegation slot: the same user turns it off
    RevokeDelegation { user: Bytes32 },
    /// a signer account: only here to sign, its shard is left alone
    Keep,
}

// ───────────────────────────── derivations (mirrored in Python) ─────────────────────────────

pub fn h(parts: &[&[u8]]) -> Bytes32 {
    let mut v = Vec::new();
    for p in parts {
        v.extend_from_slice(p);
    }
    Impl::hash_bytes(&v).as_bytes().try_into().expect("32-byte digest")
}

fn tag_seed(tag: &[u8]) -> PdaSeed {
    let mut s = [0u8; 32];
    s[..tag.len()].copy_from_slice(tag);
    PdaSeed::new(s)
}

pub fn config_seed() -> PdaSeed { tag_seed(b"neurolock/config") }
pub fn commit_seed(c: &Bytes32) -> PdaSeed { PdaSeed::new(h(&[b"nl/commit", c])) }
pub fn user_seed(u: &Bytes32) -> PdaSeed { PdaSeed::new(h(&[b"nl/user", u])) }
pub fn deleg_seed(n: &Bytes32) -> PdaSeed { PdaSeed::new(h(&[b"nl/deleg", n])) }
pub fn agent_seed(a: &Bytes32, app: &str) -> PdaSeed { PdaSeed::new(h(&[b"nl/agent", a, app.as_bytes()])) }

pub fn nullifier(user: &Bytes32, app_id: &str) -> Bytes32 {
    h(&[b"neurolock/nullifier/v1", user, app_id.as_bytes()])
}

fn pda(me: &AccountId, seed: &PdaSeed) -> AccountId { AccountId::for_public_pda(me, seed) }

fn id(a: &AccountMeta) -> Bytes32 { *a.account_id.value() }

// ───────────────────────────── plan: who may do what ─────────────────────────────

fn main() {
    run_program(plan, apply)
}

#[expect(clippy::needless_pass_by_value, reason = "run_program hands the instruction over by value")]
fn plan(input: &PlanInput, ins: Instruction) -> Plan {
    let me = input.self_account_id;
    let mut plan = Plan::new(input);
    match ins {
        Instruction::Init { party_start, party_end } => {
            let [config, admin] = input.accounts.as_slice() else { panic!("Init takes [config, admin]") };
            assert_eq!(config.account_id, pda(&me, &config_seed()), "Wrong config account");
            assert!(admin.is_authorized, "The admin must sign");
            assert!(party_start < party_end, "Empty party window");
            plan.effect(config, &Effect::InitConfig(Config { admin: id(admin), stations: vec![], party_start, party_end }));
            plan.effect(admin, &Effect::Keep);
        }
        Instruction::AddStation { station } => {
            let [config, admin] = input.accounts.as_slice() else { panic!("AddStation takes [config, admin]") };
            assert_eq!(config.account_id, pda(&me, &config_seed()), "Wrong config account");
            assert!(admin.is_authorized, "The admin must sign");
            plan.effect(config, &Effect::AdminAddStation { admin: id(admin), station });
            plan.effect(admin, &Effect::Keep);
        }
        Instruction::SetPartyWindow { party_start, party_end } => {
            let [config, admin] = input.accounts.as_slice() else { panic!("SetPartyWindow takes [config, admin]") };
            assert_eq!(config.account_id, pda(&me, &config_seed()), "Wrong config account");
            assert!(admin.is_authorized, "The admin must sign");
            assert!(party_start < party_end, "Empty party window");
            plan.effect(config, &Effect::AdminSetWindow { admin: id(admin), party_start, party_end });
            plan.effect(admin, &Effect::Keep);
        }
        Instruction::Register { commitment, user, id_pub, party_start, party_end } => {
            let [config, cslot, uslot, station] = input.accounts.as_slice() else {
                panic!("Register takes [config, commitment slot, user slot, station]")
            };
            assert_eq!(config.account_id, pda(&me, &config_seed()), "Wrong config account");
            assert_eq!(cslot.account_id, pda(&me, &commit_seed(&commitment)), "Wrong commitment slot");
            assert_eq!(uslot.account_id, pda(&me, &user_seed(&user)), "Wrong user slot");
            assert!(station.is_authorized, "An enrolment station must sign the registration");
            // The chain drops this transaction outside the claimed window; the
            // config guard checks the claim is the real window.
            let window: BlockValidityWindow = (party_start..party_end).try_into().expect("party window");
            plan.block_window(window);
            plan.effect(config, &Effect::CheckStation { station: id(station), party_start, party_end });
            let rec = Enrolled { user, id_pub };
            plan.effect(cslot, &Effect::WriteOnce(borsh::to_vec(&rec).expect("borsh")));
            plan.effect(uslot, &Effect::WriteOnce(commitment.to_vec()));
            plan.effect(station, &Effect::Keep);
        }
        Instruction::Delegate { app_id, agent, scope, expiry_block, now_block } => {
            let [uslot, dslot, aslot, user] = input.accounts.as_slice() else {
                panic!("Delegate takes [user slot, delegation slot, agent index slot, user]")
            };
            assert!(user.is_authorized, "The enrolled user must sign the delegation");
            assert!(!app_id.is_empty() && app_id.len() <= MAX_APP, "Bad app id");
            assert!(scope.len() <= MAX_SCOPE, "Scope too long");
            assert!(expiry_block > now_block, "Delegation would already be expired");
            let u = id(user);
            let n = nullifier(&u, &app_id);
            assert_eq!(uslot.account_id, pda(&me, &user_seed(&u)), "Wrong user slot");
            assert_eq!(dslot.account_id, pda(&me, &deleg_seed(&n)), "Wrong delegation slot");
            assert_eq!(aslot.account_id, pda(&me, &agent_seed(&agent, &app_id)), "Wrong agent index slot");
            // valid only once the chain has reached `now_block`, and never after expiry
            let window: BlockValidityWindow = (now_block..expiry_block).try_into().expect("delegation window");
            plan.block_window(window);
            plan.effect(uslot, &Effect::RequireEnrolled);
            let record = Delegation { nullifier: n, user: u, agent, app_id, scope, expiry_block, active: true };
            plan.effect(dslot, &Effect::PutDelegation { record, now_block });
            plan.effect(aslot, &Effect::PutIndex(n));
            plan.effect(user, &Effect::Keep);
        }
        Instruction::Revoke { app_id } => {
            let [dslot, user] = input.accounts.as_slice() else { panic!("Revoke takes [delegation slot, user]") };
            assert!(user.is_authorized, "Only the delegating user can revoke");
            let u = id(user);
            assert_eq!(dslot.account_id, pda(&me, &deleg_seed(&nullifier(&u, &app_id))), "Wrong delegation slot");
            plan.effect(dslot, &Effect::RevokeDelegation { user: u });
            plan.effect(user, &Effect::Keep);
        }
    }
    plan
}

// ───────────────────────────── apply: the rules on stored bytes ─────────────────────────────

pub fn apply(effect: Effect, pre: &[u8]) -> Option<Vec<u8>> {
    match effect {
        Effect::Keep => None,
        Effect::InitConfig(cfg) => {
            assert!(pre.is_empty(), "Registry already initialised");
            Some(borsh::to_vec(&cfg).expect("borsh"))
        }
        Effect::AdminAddStation { admin, station } => {
            let mut cfg = Config::try_from_slice(pre).expect("Registry not initialised");
            assert_eq!(cfg.admin, admin, "Only the admin can add stations");
            if !cfg.stations.contains(&station) {
                assert!(cfg.stations.len() < MAX_STATIONS, "Too many stations");
                cfg.stations.push(station);
            }
            Some(borsh::to_vec(&cfg).expect("borsh"))
        }
        Effect::AdminSetWindow { admin, party_start, party_end } => {
            let mut cfg = Config::try_from_slice(pre).expect("Registry not initialised");
            assert_eq!(cfg.admin, admin, "Only the admin can set the party window");
            cfg.party_start = party_start;
            cfg.party_end = party_end;
            Some(borsh::to_vec(&cfg).expect("borsh"))
        }
        Effect::CheckStation { station, party_start, party_end } => {
            let cfg = Config::try_from_slice(pre).expect("Registry not initialised");
            assert!(cfg.stations.contains(&station), "Unknown enrolment station");
            assert!(cfg.party_start == party_start && cfg.party_end == party_end, "Not the party window");
            None
        }
        Effect::WriteOnce(data) => {
            assert!(pre.is_empty(), "Already registered");
            Some(data)
        }
        Effect::RequireEnrolled => {
            assert!(pre.len() == 32, "Not an enrolled human");
            None
        }
        Effect::PutDelegation { record, now_block } => {
            if !pre.is_empty() {
                let old = Delegation::try_from_slice(pre).expect("delegation record");
                assert!(
                    !(old.active && old.expiry_block > now_block),
                    "This human already has an active agent for this app"
                );
            }
            Some(borsh::to_vec(&record).expect("borsh"))
        }
        Effect::PutIndex(n) => Some(n.to_vec()),
        Effect::RevokeDelegation { user } => {
            let mut d = Delegation::try_from_slice(pre).expect("No such delegation");
            assert_eq!(d.user, user, "Only the delegating user can revoke");
            d.active = false;
            Some(borsh::to_vec(&d).expect("borsh"))
        }
    }
}

/// Read-only view used by clients: active = stored, not revoked, not expired.
pub fn is_active(d: &Delegation, current_block: u64) -> bool {
    d.active && current_block < d.expiry_block
}

// ───────────────────────────── tests (run natively: `cargo test`) ─────────────────────────────
//
// The chain enforces signers and block windows before `apply` ever runs; the
// tests check those with `plan`-level helpers and the stored-state rules with `apply`.

#[cfg(test)]
mod tests {
    use super::*;

    const ADMIN: Bytes32 = [1; 32];
    const STATION: Bytes32 = [2; 32];
    const ALICE: Bytes32 = [3; 32];
    const MALLORY: Bytes32 = [4; 32];
    const AGENT1: Bytes32 = [5; 32];
    const AGENT2: Bytes32 = [6; 32];

    fn config() -> Vec<u8> {
        let c = apply(Effect::InitConfig(Config { admin: ADMIN, stations: vec![], party_start: 100, party_end: 200 }), &[]).unwrap();
        apply(Effect::AdminAddStation { admin: ADMIN, station: STATION }, &c).unwrap()
    }

    /// The chain's block-window check, as the sequencer applies it.
    fn in_window(start: u64, end: u64, block: u64) -> bool { block >= start && block < end }

    fn delegation(user: Bytes32, agent: Bytes32, app: &str, expiry: u64) -> Delegation {
        Delegation { nullifier: nullifier(&user, app), user, agent, app_id: app.into(), scope: "vote".into(), expiry_block: expiry, active: true }
    }

    #[test]
    fn valid_register() {
        let cfg = config();
        assert!(apply(Effect::CheckStation { station: STATION, party_start: 100, party_end: 200 }, &cfg).is_none());
        let rec = apply(Effect::WriteOnce(vec![9; 64]), &[]).unwrap();
        assert_eq!(rec.len(), 64);
        assert!(in_window(100, 200, 150));
    }

    #[test]
    fn register_outside_party_window_is_dropped_by_the_chain() {
        assert!(!in_window(100, 200, 99));
        assert!(!in_window(100, 200, 200));
    }

    #[test]
    #[should_panic(expected = "Not the party window")]
    fn register_claiming_a_wider_window_is_rejected() {
        apply(Effect::CheckStation { station: STATION, party_start: 0, party_end: u64::MAX }, &config());
    }

    #[test]
    #[should_panic(expected = "Unknown enrolment station")]
    fn register_with_unknown_station() {
        apply(Effect::CheckStation { station: MALLORY, party_start: 100, party_end: 200 }, &config());
    }

    #[test]
    #[should_panic(expected = "Already registered")]
    fn duplicate_commitment() {
        let first = apply(Effect::WriteOnce(vec![9; 64]), &[]).unwrap();
        apply(Effect::WriteOnce(vec![9; 64]), &first);
    }

    #[test]
    #[should_panic(expected = "Only the admin")]
    fn only_admin_adds_stations() {
        apply(Effect::AdminAddStation { admin: MALLORY, station: MALLORY }, &config());
    }

    #[test]
    fn first_delegate_accepted() {
        let d = apply(Effect::PutDelegation { record: delegation(ALICE, AGENT1, "vote", 500), now_block: 300 }, &[]).unwrap();
        assert!(is_active(&Delegation::try_from_slice(&d).unwrap(), 301));
    }

    #[test]
    #[should_panic(expected = "already has an active agent")]
    fn second_active_delegate_same_app_rejected() {
        let d = apply(Effect::PutDelegation { record: delegation(ALICE, AGENT1, "vote", 500), now_block: 300 }, &[]).unwrap();
        apply(Effect::PutDelegation { record: delegation(ALICE, AGENT2, "vote", 600), now_block: 310 }, &d);
    }

    #[test]
    fn delegate_for_a_different_app_accepted() {
        // different app -> different nullifier -> different (empty) slot
        assert_ne!(nullifier(&ALICE, "vote"), nullifier(&ALICE, "forum"));
        assert!(apply(Effect::PutDelegation { record: delegation(ALICE, AGENT2, "forum", 500), now_block: 300 }, &[]).is_some());
    }

    #[test]
    fn revoke_then_delegate_again_accepted() {
        let d = apply(Effect::PutDelegation { record: delegation(ALICE, AGENT1, "vote", 500), now_block: 300 }, &[]).unwrap();
        let r = apply(Effect::RevokeDelegation { user: ALICE }, &d).unwrap();
        assert!(!is_active(&Delegation::try_from_slice(&r).unwrap(), 301));
        assert!(apply(Effect::PutDelegation { record: delegation(ALICE, AGENT2, "vote", 600), now_block: 320 }, &r).is_some());
    }

    #[test]
    fn expired_delegation_is_inactive_and_replaceable() {
        let d = apply(Effect::PutDelegation { record: delegation(ALICE, AGENT1, "vote", 400), now_block: 300 }, &[]).unwrap();
        assert!(!is_active(&Delegation::try_from_slice(&d).unwrap(), 400));
        assert!(apply(Effect::PutDelegation { record: delegation(ALICE, AGENT2, "vote", 600), now_block: 450 }, &d).is_some());
    }

    #[test]
    #[should_panic(expected = "Only the delegating user")]
    fn forged_revoke_by_someone_else() {
        let d = apply(Effect::PutDelegation { record: delegation(ALICE, AGENT1, "vote", 500), now_block: 300 }, &[]).unwrap();
        apply(Effect::RevokeDelegation { user: MALLORY }, &d);
    }

    #[test]
    #[should_panic(expected = "Not an enrolled human")]
    fn forged_proof_unenrolled_signer_cannot_delegate() {
        // Mallory signs (the chain checks her signature) but has no user slot.
        apply(Effect::RequireEnrolled, &[]);
    }

    #[test]
    fn nullifier_is_bound_to_the_signer() {
        // Mallory can't produce Alice's nullifier: the program derives it from the signer.
        assert_ne!(nullifier(&MALLORY, "vote"), nullifier(&ALICE, "vote"));
    }
}
