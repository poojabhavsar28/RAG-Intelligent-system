#!/bin/sh
set -eu
: "${REDIS_PASSWORD:?REDIS_PASSWORD must be set}"
exec redis-server --appendonly yes --requirepass "$REDIS_PASSWORD"
