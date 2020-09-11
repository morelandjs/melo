#!/bin/bash

set -x

# install ssh private key
cd docs
openssl aes-256-cbc -K $encrypted_92725ca94bf5_key -iv $encrypted_92725ca94bf5_iv -in deploy-key.enc -out deploy-key -d
rm deploy-key.enc
chmod 600 deploy-key
mv deploy-key ~/.ssh/id_rsa

# push docs to droplet remote
if [ $TRAVIS_BRANCH == "master" ]; then
    git init

    git remote add deploy "deploy@157.230.188.74:/home/deploy/pycon"
    git config user.name "Travis CI"
    git config user.email "morelandjs@gmail.com"

    git add .
    git commit -m "deploy docs"
    git push --force deploy master
else
    echo "not deploying, since branch is not master"
fi
