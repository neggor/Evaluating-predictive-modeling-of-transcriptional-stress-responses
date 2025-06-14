source("Code/mRNA_processing/mRNA_processing_utils.R")
### HORMONE TREATMENTS
X <- read.table("Data/RAW/mRNA/hormone_treatments_raw/raw_count_matrix.txt")
X <- ceiling(X)

# Remove t=0 (Before treatment!)
pattern <- "^[A-Z]1.[0-9]+$" 
# Identify columns to drop using grepl
columns_to_drop <- grepl(pattern, names(X))
print("Keeping...")
print(names(X)[!columns_to_drop])
print("dropping...")
print(names(X)[columns_to_drop])
X <- X[, !columns_to_drop]
X <- X[!grepl("exon:", rownames(X)), ]
time_mapping <- list(
  '01' = 0.0, 
  '1' = 0.0, 
  '02' = 0.25,
  '2' = 0.25,
  '03' = 0.5,
  '3' = 0.5, 
  '05' = 1.0, 
  '5' = 1.0, 
  '07' = 1.5, 
  '7' = 1.5, 
  '09' = 2.0, 
  '9' = 2.0, 
  '13' = 3.0, 
  '15' = 4.0, 
  '17' = 5.0, 
  '19' = 6.0, 
  '21' = 7.0, 
  '23' = 8.0, 
  '26' = 10.0, 
  '28' = 12.0, 
  '30' = 16.0
)

# Format data to fit the general structure
convert_colnames <- function(label) {
  # Use gsub to transform the format and replace "A" with "control"
  modified_label <- gsub("(\\D+)(\\d+)\\.(\\d+)", "\\1_\\2_\\3", label)
  # Replace "A_" with "control_" in the modified label
  modified_label <- gsub("^A_", "control_", modified_label)
  key <- gsub(".*_(\\d+)_.*", "\\1", modified_label)
  value <- time_mapping[[key]]
  modified_label <- gsub(paste0("_(\\d+)_"), paste0("_", value, "_"), modified_label)
}

# Apply the function to all column names
new_colnames <- sapply(colnames(X), convert_colnames)

# Assign the new column names to your array
colnames(X) <- new_colnames

# drop columns with time >= 3h
selected_colnames <- grep("_(0|0.25|0.5|1|1.5|2|3)_", new_colnames, value = TRUE)
X <- X[selected_colnames]

convert_to_minutes <- function(colname) {
  # Extract the part between the second "_" and the end
  parts <- strsplit(colname, "_")[[1]]
  
  # If there is a decimal (e.g., 0.25, 0.5), multiply by 60 to get minutes
  if (grepl("\\.", parts[2])) {
    parts[2] <- as.character(as.numeric(parts[2]) * 60)
  } else {
    parts[2] <- as.character(as.numeric(parts[2]) * 60)
  }
  
  # Recombine the parts
  paste(parts, collapse = "_")
}

# Apply the conversion function to all column names
converted_colnames <- sapply(colnames(X), convert_to_minutes)
colnames(X) <- converted_colnames
print(colnames(X))



final_results <- DE_analysis(X, treatments = c("C"), store_folder = "Data/Processed/mRNA", name = "Hormone")
write.csv(final_results, "Data/Processed/mRNA/DESeq2_padj_results_Hormone.csv", row.names = FALSE)


#### PTI
rm() # clean the environment, just in case...

X = read.csv("Data/RAW/mRNA/PTI_raw/RAW_COUNTS_PTI.csv",row.names = 1)
# drop the time point 0
X <- X[, !grepl('_000_', colnames(X))]
final_results <- DE_analysis(X, treatments = c("X", "Y", "Z", "W", "V", "U", "T"), store_folder = "Data/Processed/mRNA", name = "PTI")
write.csv(final_results, "Data/Processed/mRNA/DESeq2_padj_results_PTI.csv", row.names = FALSE)


