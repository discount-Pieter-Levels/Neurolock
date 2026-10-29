#!/bin/bash
# Wait for the LEZ testnet to produce blocks steadily, then deploy + evaluate once.
# Healthy = >= 8 new blocks in a 5-minute window (normal block time ~30 s).
cd "$(dirname "$0")/.."
RPC=https://testnet.lez.logos.co
blk() { curl -s -m 20 -X POST $RPC -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"getLastBlockId","params":[]}' | grep -o '"result":[0-9]*' | cut -d: -f2; }
while true; do
  a=$(blk); sleep 300; b=$(blk)
  echo "$(date +%T) blocks $a -> $b"
  [ -n "$a" ] && [ -n "$b" ] && [ $((b - a)) -ge 8 ] && break
done
echo "$(date +%T) testnet healthy, deploying"
PY=logos_build/nixpy.sh
$PY logos/deploy_registry.py deploy 2>&1 | grep -v '^Transaction data' > logos_build/deploy.log || { echo "DEPLOY FAILED"; tail -5 logos_build/deploy.log; exit 1; }
tail -5 logos_build/deploy.log
$PY logos/deploy_registry.py status 2>&1 | grep -v '^Transaction data' | tail -15
echo "$(date +%T) running demo on chain"
$PY run_pop_demo.py --ledger logos 2>&1 | grep -v '^Transaction data' > logos_build/demo_logos.log; echo "demo exit $?"
echo "$(date +%T) running evaluate_logos"
$PY logos/evaluate_logos.py --people 10 2>&1 | grep -v '^Transaction data' > logos_build/evaluate_logos.log; echo "evaluate exit $?"
echo "ALL DONE $(date +%T)"
