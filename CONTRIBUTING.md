Only merge feature branches into main. Do not merge python and cpp branches directly into each other.

```treminal
# On python branch
git checkout python
git add .
git commit -m "Python feature"
git checkout main
git pull origin main
git merge python
git push origin main

# On cpp branch
git checkout cpp
git add .
git commit -m "C++ feature"
git checkout main
git pull origin main
git merge cpp
git push origin main
```
