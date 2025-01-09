#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print with color
print_message() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

# Get the commit message
if [ -z "$1" ]
then
    print_warning "Please provide a commit message!"
    read -p "Commit message: " commit_message
else
    commit_message=$1
fi

print_message "\n🚀 Starting MediPredictPro update process..."

# Add all changes
print_message "\n📦 Adding changes..."
git add .

# Commit changes
print_message "\n💬 Committing with message: $commit_message"
git commit -m "$commit_message"

# Pull latest changes
print_message "\n⬇️ Pulling latest changes..."
git pull origin main

# Push changes
print_message "\n⬆️ Pushing changes..."
git push origin main

print_message "\n✅ Update complete!"
