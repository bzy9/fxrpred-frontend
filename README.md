# FXRPred frontend

Streamlit website for FACA-GIN, Ortho-MoE, benchmark resources and help.
Entry point: app.py. Model inference is served separately by
gzLiang/fxr-model-service on Hugging Face ZeroGPU.

## Publish

Upload this directory's CONTENTS to a GitHub frontend repository root. Preserve
assets/, data/benchmarks/ and .streamlit/secrets.toml.example. Do not include
the outer frontend_release folder, weights, local IDE files or real secrets.
On Streamlit Community Cloud select the repository, main branch and app.py.
Choose Python 3.11 in Advanced settings, and enter your own Secrets there:

```toml
HF_SPACE_ID = "gzLiang/fxr-model-service"
HF_TOKEN = "hf_your_actual_token"
FACA_API_BATCH_SIZE = 16
ORTHO_API_BATCH_SIZE = 4
```

Token values use ASCII quotes. Never upload a real secrets.toml or token.
Environment variables take precedence over Streamlit Secrets. The client
authenticates with the account owning the token; visitors share its quota.
Model storage being public does not remove ZeroGPU authentication/quota limits.

## Sequential batches

No fixed total record cap is applied by this client. The matching updated
backend defaults MAX_INPUT_RECORDS to 0 (no optional record cap). Upload the
updated HF backend app.py before release. If a Space Variable still sets
MAX_INPUT_RECORDS=100, remove it or set it to 0.

FACA submits 16 records per API request and Ortho submits 4 by default, with a
visible progress bar. Batch size can be configured in Secrets. Model-internal
batch settings remain independent; do not raise request batches without timing
checks. There are still GPU time, run-count, memory and upload-size limits.

All results are validated before a batch is committed. Input order and repeated
records are retained. Ortho np_id strings (including leading zeros) stay aligned
with the corresponding input rows. FACA retains backend omission of invalid
structures and shows the skipped count. An all-invalid FACA batch may stop with
the backend's error, with prior results retained.

If a batch fails, no automatic retries or CPU fallback are attempted. The page
retains completed batches in the CURRENT Streamlit session and offers completed
results and remaining-input CSV downloads. Fix the error or wait for quota, then
upload the remaining-input CSV as a new request. Downloads do not rerun inference.
Closing the session, restarting the app, or changing its input may discard the
session's results; this is not a durable job database.

FACA CSV includes Input_record: a 1-based index into the submitted nonempty SMILES
list. Blank source rows are filtered before that numbering. Its attention panel
keeps at most three active images across completed batches, ranked by the
backend-returned four-decimal probability; ties use later input index first.
This ranking does not change probabilities, classifications or thresholds.
The Ortho probabilities retain their original returned precision.

## Dependencies and resources

Direct dependencies are pinned to versions read from the working local frontend
environment's package metadata, without running that environment or changing it.
Unused frontend RDKit is omitted: current pages use precomputed descriptor data
and backend-rendered molecular images. The backend's RDKit and model dependencies
are unchanged. This is not a full lock of every transitive dependency.

Only benchmark CSV files, descriptor cache and its manifest are included under
data/. Historical application-one screening results are not part of the site.
Online preview retains source IDs; downloadable benchmark CSV columns remain
defined by data_resources.py. Home, Help and Data Resources do not run models.

## Verification status

Files were prepared and checked offline. No local web server, model inference,
package installation, token inspection or HF/GitHub upload was performed.
Model-specific and earlier formal backend stages passed on HF by user report;
new batching UI and Cloud packaging still need online validation. Start with
one molecule and then enough inputs to cross a batch boundary (17 FACA or 5
Ortho), check result order/downloads, and test interruption/recovery when useful.
