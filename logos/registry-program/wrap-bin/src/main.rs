//! `cargo run --release -- <user.elf> <out.bin>`
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let user = std::fs::read(&a[1]).expect("read user ELF");
    let bin = risc0_binfmt::ProgramBinary::new(&user, risc0_zkos_v1compat::V1COMPAT_ELF).encode();
    let id = risc0_binfmt::compute_image_id(&bin).expect("image id");
    std::fs::write(&a[2], &bin).expect("write .bin");
    println!("wrote {} ({} bytes), image id {}", a[2], bin.len(), id);
}
