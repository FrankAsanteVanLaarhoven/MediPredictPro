#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Setting up GitHub Authentication...${NC}"

# Configure Git user
echo -e "${YELLOW}Enter your GitHub username:${NC}"
read GITHUB_USERNAME
git config --global user.name "$GITHUB_USERNAME"

echo -e "${YELLOW}Enter your GitHub email:${NC}"
read GITHUB_EMAIL
git config --global user.email "$GITHUB_EMAIL"

# Instructions for creating Personal Access Token
echo -e "${YELLOW}Please create a Personal Access Token (PAT) on GitHub:${NC}"
echo "1. Go to GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)"
echo "2. Click 'Generate new token' → 'Generate new token (classic)'"
echo "3. Name: MediPredictPro"
echo "4. Select scopes: repo, workflow"
echo "5. Click 'Generate token'"
echo "6. Copy the token (you'll only see it once!)"
echo
echo -e "${YELLOW}Enter your Personal Access Token:${NC}"
read -s GITHUB_TOKEN

# Store credentials
echo -e "https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com" > ~/.git-credentials
git config --global credential.helper store

echo -e "${GREEN}Authentication setup complete!${NC}"
