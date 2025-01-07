#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Personal Information
FULL_NAME="Frank Asante Van Laarhoven"
GITHUB_USERNAME="FAVL"
REPO_NAME="MediPredictPro"
EMAIL="your.email@example.com"  # Replace with your email

echo -e "${YELLOW}Setting up GitHub repository for $FULL_NAME...${NC}"

# Initialize repository
if [ ! -d .git ]; then
    git init
    echo -e "${GREEN}Git repository initialized${NC}"
fi

# Create main branch
git checkout -b main 2>/dev/null || git checkout main

# Add files
git add .

# Commit
git commit -m "Initial commit by $FULL_NAME"

# Get GitHub repository URL
echo -e "${YELLOW}Please enter your GitHub repository URL:${NC}"
read GITHUB_URL

# Add remote
git remote add origin $GITHUB_URL

# Push to GitHub
echo -e "${YELLOW}Pushing to GitHub...${NC}"
git push -u origin main
