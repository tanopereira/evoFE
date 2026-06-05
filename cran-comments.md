## Test environments
* local macOS 15.4 (arm64), R 4.6.0
* win-builder (devel and release)
* R-hub v2:
  - Linux (ubuntu-latest), R-release and R-devel
  - Windows (windows-latest), R-release and R-devel
  - macOS (macos-13 and macos-15-arm64), R-release and R-devel

## R CMD check results

0 errors | 0 warnings | 2 notes

* This is a resubmission. In response to CRAN reviewer feedback, the following issues have been resolved:
  - DESCRIPTION: Expanded the Description text to provide more details about the package's functionality and implemented methods.
  - DESCRIPTION: Added references for methods used by the package (McInnes et al. 2018 for UMAP, Ke et al. 2017 for LightGBM, Chen & Guestrin 2016 for XGBoost, Gagolewski 2021 for genieclust, Gagolewski 2026 for lumbermark and deadwood) in the correct format (`<doi:...>`, `<https:...>`).
  - Spelling: Added `inst/WORDLIST` to whitelist proper names and technical acronyms (Gagolewski, Guestrin, Ke, McInnes, UMAP, et, al) to resolve the spelling note.
  - Documentation: Added detailed `\value` tags to all 17 exported functions' `.Rd` files to explain function results and their structure.
  - Reproducibility: Removed all `set.seed()` calls inside the functions (`evolve_features` and transformers) to avoid altering the user's workspace seed. Updated tests and vignettes to demonstrate how to set seed reproducibility externally.
  - Vignette CPU time: The CPU time note has been resolved by reducing the default thread count to 2.

* Remaining notes:
  - The standard `New submission` note from `CRAN incoming feasibility`.
  - HTML Tidy note (local HTML Tidy too old, not a package issue).
