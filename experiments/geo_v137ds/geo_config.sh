#!/usr/bin/env bash
# v137ds -- the pinned GEO configuration.
#
# Planner gpt-5.4-mini + Writer deepseek-v4-flash.  Measured on camera_product at
# N=150 (docs/DECISIONS.md G154-G175):
#
#     PASS 8/12   self_bertscore d +0.10 p 0.143 PASS   self_bleu_4 d -0.03 p 0.610 PASS
#     fails: hard_disagree (KS 0.011), impolite (0.000), length_cv (0.032), emotion_entropy (KS 0.005)
#
# Every flag below is part of the pin.  `--length-ceiling` is DELIBERATELY absent
# so it defaults to off: v138 priced that arm and it was rejected (G167/G168).
# Do not add, remove or reorder a flag without a new version name.
#
# Sourced by run_geo_domain.sh.  Not executable on its own.

GEO_V137DS_FLAGS=(
  --closing-move measured --context-dropout-rate 0.42 --context-jitter-rate 0.32
  --development-scope measured --digit-cue-guard off --domain-claim selective
  --downtoner-tag suppress --entity-spread off --evaluation-tier measured
  --final-punctuation measured --interaction-scope off --length-calibration measured
  --length-fidelity off --length-transfer v97 --long-form-layout measured
  --no-story-scope sequence --opening-move measured --outsider-quota off
  --own-fact-license off --partitive-reference suppress --plan-move-ledger off
  --reddit-typography on
  --reference-link measured --reference-link-count measured --reference-link-host off
  --register-realization measured --reply-sibling-visibility on
  --rhythm-count measured --route-ledger on --semantic-coverage-nonrepeat on
  --sentence-pacing off --sentence-rhythm measured --social-contract-coherence on
  --speaker-identity matched --tone-donor off --tone-length-fit conditional
  --tone-quota inverted --turn-frame adjudicative_only --verdict-close-guard off
  --writer-prompt focused --writer-retries 0 --writer-route-lock own_words
  --recurring-phrase-ledger off --post-retry-limit 3
)

# Planner is fixed; only the Writer varies across the model sweep.
GEO_V137DS_PLANNER="gpt-5.4-mini"

# Every domain with a config under generalized_card/configs/domains/.  The four
# originals plus whatever enable_domain.sh has added (those carry a _geo suffix).
# A domain still needs scored real threads before it can be EVALUATED; it can be
# generated as soon as the config exists.
GEO_V137DS_DOMAINS="$(ls "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/generalized_card/configs/domains"/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//' | tr '\n' ' ')"

# Seed pool per domain: "<pool-size> <sampling-seed>".  These name EXISTING files
# under artifacts/generalized_card/seed_pools/.  run_generate REBUILDS a missing
# pool from (size, seed) and a rebuild does NOT reproduce the original sample
# (G165) -- run_geo_domain.sh refuses to start if the file is gone.
geo_pool_for() {
  case "$1" in
    camera)     echo "150 907" ;;
    headphone)  echo "150 42"  ;;
    cell_phone) echo "100 42"  ;;
    laptop)     echo "100 42"  ;;
    # A domain enabled by enable_domain.sh has no pool yet; run_generate builds
    # one at (size, seed) on first use, and from then on the file is the pin.
    *_geo)      echo "150 907" ;;
    *)          echo ""        ;;
  esac
}

geo_domain_id() {
  case "$1" in
    camera)     echo "camera_product" ;;
    cell_phone) echo "cell_phone_product" ;;
    headphone)  echo "headphone_product" ;;
    laptop)     echo "laptop_product" ;;
    # enable_domain.sh writes configs/domains/<name>_geo.json with domain_id == <name>_geo
    *_geo)      echo "$1" ;;
    *)          echo "" ;;
  esac
}

# "<base-url> <key-env>" for any model, used for the Planner and the Writer
# alike -- v137ds pinned Planner=gpt-5.4-mini, but a same-model arm points both
# ends at one provider.  Key names are the ones in third_party/MiroFish/.env.
geo_model_endpoint() {
  case "$1" in
    deepseek-v4-flash|deepseek-v4-pro|deepseek-chat)
      echo "https://api.deepseek.com/v1 deepseek_api_key" ;;
    gpt-5.4-mini|gpt-4o-mini|gpt-4.1-mini)
      echo "https://api.openai.com/v1 LLM_API_KEY" ;;
    gemini-2.5-flash|gemini-2.5-pro)
      echo "https://generativelanguage.googleapis.com/v1beta/openai/ gemini_api_key" ;;
    *)  echo "" ;;
  esac
}
geo_writer_endpoint() { geo_model_endpoint "$1"; }

# Short form used in run tags and log paths.
geo_model_short() {
  echo "$1" | sed 's/deepseek-v4-flash/dsflash/; s/deepseek-v4-pro/dspro/;
                   s/gpt-5.4-mini/g54m/; s/gpt-4o-mini/g4om/;
                   s/gemini-2.5-flash/gem25f/; s/[^a-z0-9]//g'
}

# Provider concurrency. DeepSeek v4-flash allows 2500 concurrent; the others do
# not, so a same-model arm inherits the SLOWER of its two ends.
geo_default_parallel() {
  case "$1" in
    deepseek-*) echo 50 ;;
    gemini-*)   echo 8 ;;
    *)          echo 10 ;;
  esac
}
