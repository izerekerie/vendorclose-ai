# 🚀 Using Docker Cache - Faster Builds

## What I Changed:

1. **Split requirements.txt** into two files:

   - `requirements-base.txt` - Heavy packages (TensorFlow) - cached separately
   - `requirements-light.txt` - Light packages - cached separately

2. **Optimized Dockerfile** - Installs heavy packages first, so they're cached

3. **Added .dockerignore** - Excludes large files from build context (faster transfers)

## How Docker Cache Works:

**First build (slow - 5-10 minutes):**

```bash
docker-compose build api
```

**Next builds (FAST - uses cache!):**

```bash
# If you only change code (not requirements), TensorFlow layer is cached!
docker-compose build api  # Only rebuilds changed layers

# Starting other containers uses SAME image (instant!)
docker-compose up -d api-2  # Starts in seconds!
docker-compose up -d api-3  # Starts in seconds!
```

## Check if Image Exists (Use Cache):

```bash
# Check existing images
docker images | grep summative_pipeliines

# If image exists, rebuild will use cache
docker-compose build api
```

## Force Fresh Build (No Cache):

```bash
docker-compose build --no-cache api
```

## Tips:

1. **Don't change requirements.txt often** - changes invalidate cache
2. **Code changes are fast** - only rebuilds last layer
3. **First build is slow** - but subsequent builds are fast!
4. **Multiple containers share image** - `api-2` and `api-3` start instantly after first build

## Current Setup:

- ✅ TensorFlow installs in separate layer (cached)
- ✅ Light packages in separate layer (cached)
- ✅ Code copied last (changes most often)
- ✅ Large files excluded from build context

**Result:** After first build, code changes rebuild in ~30 seconds instead of 10 minutes!
