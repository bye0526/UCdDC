# --- 1) Read data from Excel (robust to .xlxs/.xlsx typo) ---
candidate_files <- c("ML2025.xlsx", "ML2025.xlxs")
data_file <- candidate_files[file.exists(candidate_files)][1]
if (is.na(data_file)) stop("Could not find ML2025.xlsx (or ML2025.xlxs) in the working directory.")

sheets <- readxl::excel_sheets(data_file)
preferred <- c("SEM", "sem", "data", "Data", "sheet1")
sheet_to_read <- if (any(preferred %in% sheets)) preferred[preferred %in% sheets][1] else sheets[1]
raw <- readxl::read_excel(path = data_file, sheet = sheet_to_read)

# --- 2) Basic column checks and tolerant renaming ---
# Add diet subpaths (solanaceous/legumes/root_tuber/other)
std_names <- c(
  "HID", "Urine_Cd_CREA", "Smoke",
  "BMI", "gender", "Renal_Dysfunction", "Hypertension", "Diabetes",
  "age", "Distance_km",
  "soil_Cd", "rice_Cd",
  "soil_dermal", "soil_ingestion",
  "diet_total", "env_total",
  "diet_rice", "diet_solanaceous", "diet_legumes", "diet_root_tuber", "diet_other",
  "water_ingestion", "Water_dermal"
)

cn <- colnames(raw)
map_name <- function(target) {
  hit <- which(tolower(cn) == tolower(target))
  if (length(hit) == 1) return(cn[hit])
  fallback <- switch(tolower(target),
                     "urine_cd_crea"   = c("urine_cd", "urine_cd_crea_y", "urine_cd_creatinine"),
                     "smoke"           = c("smoking"),
                     "distance_km"     = c("distance", "dist_km", "distancekm"),
                     "soil_cd"         = c("soil_cadmium", "soilcd", "soil_cd_mgkg"),
                     "diet_rice"       = c("diet_rice_intake", "rice_exposure", "dietrice"),
                     "diet_solanaceous"= c("diet_solan", "diet_sola", "diet_solanaceae"),
                     "diet_legumes"    = c("diet_legume", "diet_pulses"),
                     "diet_root_tuber" = c("diet_root", "diet_tuber", "diet_roottuber"),
                     "diet_other"      = c("diet_misc", "diet_others"),
                     "water_ingestion" = c("water_intake_exposure", "water_exposure"),
                     target
  )
  for (fb in fallback) {
    hit2 <- which(tolower(cn) == tolower(fb))
    if (length(hit2) == 1) return(cn[hit2])
  }
  return(NA_character_)
}

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

# --- 3) Build SEM analysis dataset (scaling, transforms) ---
eps_cd <- 1e-3
sem_data <- dat %>%
  mutate(
    log_Urine_Cd = log(Urine_Cd_CREA + eps_cd),
    smoking = Smoke,
    age_sc = as.numeric(scale(age)),
    distance_sc = as.numeric(scale(Distance_km)),
    
    # media Cd (scaled)
    soil_Cd_scaled         = as.numeric(scale(soil_Cd)),
    rice_Cd_scaled         = as.numeric(scale(rice_Cd)),
    
    # exposure subpaths (scaled)
    soil_dermal_scaled     = as.numeric(scale(soil_dermal)),
    soil_ingestion_scaled  = as.numeric(scale(soil_ingestion)),
    water_ingestion_scaled = as.numeric(scale(water_ingestion)),
    Water_dermal_scaled    = as.numeric(scale(Water_dermal)),
    
    # diet subpaths (scaled) — NEW
    diet_rice_scaled        = as.numeric(scale(diet_rice)),
    diet_solanaceous_scaled = as.numeric(scale(diet_solanaceous)),
    diet_legumes_scaled     = as.numeric(scale(diet_legumes)),
    diet_root_tuber_scaled  = as.numeric(scale(diet_root_tuber)),
    diet_other_scaled       = as.numeric(scale(diet_other)),
    
    # totals (optional; kept for reference/diagnostics)
    diet_total_scaled      = as.numeric(scale(diet_total)),
    env_total_scaled       = as.numeric(scale(env_total))
  ) %>%
  dplyr::select(
    HID, BMI, gender, age_sc, Renal_Dysfunction, Hypertension, Diabetes,
    log_Urine_Cd, smoking,
    soil_Cd_scaled, rice_Cd_scaled,
    soil_dermal_scaled, soil_ingestion_scaled, Water_dermal_scaled,
    water_ingestion_scaled,
    diet_rice_scaled, diet_solanaceous_scaled, diet_legumes_scaled,
    diet_root_tuber_scaled, diet_other_scaled,
    diet_total_scaled, env_total_scaled,
    distance_sc
  ) %>%
  tidyr::drop_na()


# --- 4) Specify SEM (latent Exposure with two indicators) ---
model_sem <- '
  # Latent variable: overall environmental exposure
  #Exposure =~ soil_ingestion_scaled + diet_rice_scaled 
  #Exposure =~  diet_total_scaled + env_total_scaled
  # Upstream determinants of indicators
  #soil_ingestion_scaled ~ soil_Cd_scaled
  #diet_rice_scaled      ~ rice_Cd_scaled

  # Distance effects
  #Exposure      ~ distance_sc
  #soil_Cd_scaled ~ distance_sc
  #rice_Cd_scaled ~ distance_sc
  #rice_Cd_scaled ~ soil_Cd_scaled

  # Biomonitoring outcome
  log_Urine_Cd ~ soil_ingestion_scaled + diet_rice_scaled +Water_dermal_scaled + 
                 diet_solanaceous_scaled +diet_legumes_scaled+ gender + BMI + age_sc
                 

  # Health outcomes
  Hypertension       ~ log_Urine_Cd + age_sc + gender + BMI
  #Renal_Dysfunction  ~ log_Urine_Cd + Hypertension + Diabetes + gender + BMI
  Diabetes           ~ log_Urine_Cd + age_sc + gender + BMI
'

# --- 5) Fit SEM with robust ML (MLR) ---
fit_sem <- lavaan::sem(
  model = model_sem,
  data  = sem_data,
  estimator = "WLSMV",
  ordered = c("Hypertension", "Diabetes"),  # 确保是 0/1 或有序因子
  std.lv = TRUE
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
