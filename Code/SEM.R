# ============================================================
# SEM pipeline (reproducible): read ML2025.xlsx and run lavaan
# ============================================================

# --- Packages ---
req_pkgs <- c("readxl", "dplyr", "tidyr", "lavaan", "semPlot", "openxlsx")
new_pkgs <- req_pkgs[!(req_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs, dependencies = TRUE)

library(readxl)
library(dplyr)
library(tidyr)
library(lavaan)
library(semPlot)
library(openxlsx)

# --- 1) Read data from Excel (robust to the .xlxs/.xlsx typo) ---
candidate_files <- c("ML2025.xlsx", "ML2025.xlxs")
data_file <- candidate_files[file.exists(candidate_files)][1]
if (is.na(data_file)) stop("Could not find ML2025.xlsx (or ML2025.xlxs) in the working directory.")

sheets <- readxl::excel_sheets(data_file)
# Prefer a sheet named "SEM" or "data"; otherwise, use the first sheet
preferred <- c("SEM", "sem", "data", "Data", "sheet1")
sheet_to_read <- if (any(preferred %in% sheets)) preferred[preferred %in% sheets][1] else sheets[1]

raw <- readxl::read_excel(path = data_file, sheet = sheet_to_read)

# --- 2) Basic column checks and minimal renaming ---
# Expect these columns to exist in ML2025.xlsx (snake_case is fine; case-insensitive)
# If your columns already match, this check will pass and no renaming is needed.
std_names <- c(
  "HID", "Urine_Cd_CREA",  "Smoke",
  "BMI", "gender", "Renal_Dysfunction", "Hypertension", "Diabetes",
  "age", "Distance_km", "soil_Cd","rice_Cd", "soil_dermal", "soil_ingestion",
  "diet_total", "env_total", "diet_rice", "water_ingestion","Water_dermal"
)

# Create a case-insensitive lookup to tolerate minor name variations
cn <- colnames(raw)
map_name <- function(target) {
  hit <- which(tolower(cn) == tolower(target))
  if (length(hit) == 1) return(cn[hit])
  # common fallbacks (add more if needed)
  fallback <- switch(tolower(target),
                     "urine_cd_crea"       = c("urine_cd", "urine_cd_crea_y"),
                    
                     "smoke"               = c("smoking"),
                     "distance_km"         = c("distance", "dist_km", "distancekm"),
                     "soil_cd"             = c("soil_cadmium", "soilcd", "soil_cd_mgkg"),
                     "diet_rice"           = c("diet_rice_intake", "rice_exposure", "dietrice"),
                     "water_ingestion"     = c("water_intake_exposure", "water_exposure"),
                     target
  )
  for (fb in fallback) {
    hit2 <- which(tolower(cn) == tolower(fb))
    if (length(hit2) == 1) return(cn[hit2])
  }
  return(NA_character_)
}

# Build a rename map for the variables we need
rename_map <- setNames(nm = std_names, object = vapply(std_names, map_name, character(1)))
missing_cols <- names(rename_map)[is.na(rename_map)]
if (length(missing_cols)) {
  stop(
    "Missing required columns in sheet '", sheet_to_read, "': ",
    paste(missing_cols, collapse = ", "),
    "\nAvailable columns are: ", paste(colnames(raw), collapse = ", ")
  )
}

dat <- raw %>%
  dplyr::rename(!!!setNames(rename_map[!is.na(rename_map)], names(rename_map)[!is.na(rename_map)]))

# --- 3) Build SEM analysis dataset (transformations) ---
# Notes:
# - Log transforms add small epsilons to avoid log(0).
# - Standardization uses scale() and coerces to numeric vector.
# - Outcomes Hypertension / Diabetes / Renal_Dysfunction are kept numeric as-is.
eps_cd  <- 1e-3
eps_se  <- 5e-2

sem_data <- dat %>%
  mutate(
    log_Urine_Cd = log(Urine_Cd_CREA + eps_cd),
   
    
    # Keep continuous/binary covariates as provided in the file
    
    smoking = Smoke,
    
    # Standardize age and distance
    age_sc = as.numeric(scale(age)),
    distance_sc = as.numeric(scale(Distance_km)),
    
    # Standardize exposure variables
    soil_Cd_scaled         = as.numeric(scale(soil_Cd)),
    rice_Cd_scaled         = as.numeric(scale(rice_Cd)),
    soil_dermal_scaled     = as.numeric(scale(soil_dermal)),
    soil_ingestion_scaled  = as.numeric(scale(soil_ingestion)),
    diet_total_scaled      = as.numeric(scale(diet_total)),
    env_total_scaled       = as.numeric(scale(env_total)),
    diet_rice_scaled       = as.numeric(scale(diet_rice)),
    Water_dermal_scaled = as.numeric(scale(Water_dermal)),
    water_ingestion_scaled = as.numeric(scale(water_ingestion))
  ) %>%
  dplyr::select(
    HID, BMI, gender, age_sc, Renal_Dysfunction, Hypertension, Diabetes,
    log_Urine_Cd, 
    smoking,
    soil_Cd_scaled, rice_Cd_scaled, soil_dermal_scaled, soil_ingestion_scaled, Water_dermal_scaled,
    diet_total_scaled, env_total_scaled, diet_rice_scaled, water_ingestion_scaled,
    distance_sc
  ) %>%
  tidyr::drop_na()

# --- 4) Specify SEM (latent Exposure with two indicators) ---
model_sem <- '
  # Latent variable: overall environmental exposure
  Exposure =~ soil_ingestion_scaled + diet_rice_scaled

  # Upstream determinants of indicators
  soil_ingestion_scaled ~ soil_Cd_scaled
  #diet_rice_scaled      ~ rice_Cd_scaled

  # Distance effects
  Exposure      ~ distance_sc
  #soil_Cd_scaled ~ distance_sc
  #rice_Cd_scaled ~ distance_sc
  #rice_Cd_scaled ~ soil_Cd_scaled

  # Biomonitoring outcome
  log_Urine_Cd ~ Exposure + gender + BMI + age_sc

  # Health outcomes
  Hypertension       ~ log_Urine_Cd + age_sc + gender + BMI
  #Renal_Dysfunction  ~ log_Urine_Cd + Hypertension + Diabetes + gender + BMI
  Diabetes           ~ log_Urine_Cd + age_sc + gender + BMI
'

# --- 5) Fit SEM with robust ML (MLR) ---
fit_sem <- lavaan::sem(
  model = model_sem,
  data  = sem_data,
  estimator = "MLR",
  std.lv=TRUE,
  missing   = "ml"
)

# --- 6) Summaries and key fit measures ---
cat("\n=== SEM Summary (standardized) ===\n")
print(summary(fit_sem, fit.measures = TRUE, standardized = TRUE))

fit_keys <- fitMeasures(
  fit_sem,
  fit.measures = c("cfi.robust","tli.robust","rmsea.robust","srmr_bentler",
                   "chisq.scaled","df.scaled","pvalue.scaled")
)
cat("\n=== Key Fit Measures (robust/scaled) ===\n")
print(fit_keys)

# Parameter table with significance stars
para_tbl <- parameterEstimates(fit_sem, standardized = TRUE) %>%
  dplyr::filter(op %in% c("=~", "~", "~~")) %>%
  dplyr::mutate(
    sig = dplyr::case_when(
      pvalue < 0.001 ~ "***",
      pvalue < 0.01  ~ "**",
      pvalue < 0.05  ~ "*",
      TRUE           ~ ""
    )
  ) %>%
  dplyr::select(lhs, op, rhs, est, se, z, std.all, pvalue, sig)

cat("\n=== Parameter Estimates (standardized) ===\n")
print(para_tbl)

# --- 7) Modification indices (optional diagnostics) ---
mi_tbl <- modificationIndices(fit_sem, sort. = TRUE, minimum.value = 3.84)
cat("\n=== Top Modification Indices (MI >= 3.84) ===\n")
print(head(mi_tbl, 10))

# --- 8) Path diagram (TIFF) ---
tiff("SEM_Revised_Paths.tiff", width = 2400, height = 1800, res = 300)
semPlot::semPaths(
  fit_sem,
  what = "std",
  whatLabels = "std",
  layout = "tree2",
  style = "lisrel",
  edge.label.cex = 0.9,
  sizeMan = 10,
  sizeLat = 12,
  nCharNodes = 0,
  residuals = TRUE,
  title = FALSE,
  mar = c(8, 2, 4, 2)
)
mtext(
  paste0(
    "χ²(", round(fit_keys["df.scaled"]), ") = ", round(fit_keys["chisq.scaled"], 2),
    ", CFI = ", round(fit_keys["cfi.robust"], 3),
    ", RMSEA = ", round(fit_keys["rmsea.robust"], 3),
    ", SRMR = ", round(fit_keys["srmr_bentler"], 3)
  ),
  side = 1, line = 5.5, cex = 0.9
)
dev.off()
cat("\nSaved path diagram: SEM_Revised_Paths.tiff\n")

# --- 9) Export results to Excel ---
wb <- openxlsx::createWorkbook()

openxlsx::addWorksheet(wb, "FitMeasures")
openxlsx::writeData(wb, "FitMeasures", as.data.frame(t(fit_keys)))

openxlsx::addWorksheet(wb, "Parameters")
openxlsx::writeData(wb, "Parameters", para_tbl)

if (nrow(mi_tbl) > 0) {
  openxlsx::addWorksheet(wb, "ModIndices_Top50")
  openxlsx::writeData(wb, "ModIndices_Top50", head(mi_tbl, 50))
}

openxlsx::saveWorkbook(wb, "SEM_Revised_Results.xlsx", overwrite = TRUE)
cat("Exported results: SEM_Revised_Results.xlsx\n")

# --- 10) Notes for reviewers (printed to console) ---
cat("\nGuideline targets: CFI/TLI > 0.90, RMSEA < 0.08, SRMR < 0.08.\n")
