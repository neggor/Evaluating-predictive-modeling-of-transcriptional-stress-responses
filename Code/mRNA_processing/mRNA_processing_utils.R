library(DESeq2)
library(dplyr)
library(ggplot2)
library(tidyr)
library(stringr)
library(splines)
library(pheatmap)
library(reshape2)
library(clustra)
library(tibble)
library(gridExtra)
library(grid)

#These functions expect the column names to be <treatment>_<time>_<replicate>
#Control is needed always, and as such it must follow the name control_<time>_<replicate>

create_subset_matrix <- function(X, treatment, time_as_numeric = FALSE) {
  
  # Extract column names
  columns <- colnames(X)
  # Identify columns that match the treatment and time point
  control_columns <- grep("^control", columns, value = TRUE)
  treatment_columns <- grep(paste0("^", treatment), columns, value = TRUE)
  
  # Combine control and treatment columns
  selected_columns <- c(control_columns, treatment_columns)
  print("Selected columns:")
  print(selected_columns)
  # Subset the data
  subset_matrix <- X[, selected_columns]
  
  # Pattern to extract time:
  pattern_time <- "(\\d+(\\.\\d+)?)"
  extracted_times <- str_extract(selected_columns, pattern_time) # Is going to be treated like a factor!
  if (time_as_numeric){
    extracted_times <- as.numeric(paste0(extracted_times))
  }
  
  # Extract the treatment
  treatment_labels <- c(rep("control", length(control_columns)), rep(treatment, length(treatment_columns)))
  factorized_treatment_labels <- factor(treatment_labels, levels = c("control", treatment))
  
  coldata <- data.frame(row.names = selected_columns, Treatment = factorized_treatment_labels, Time = extracted_times)
  # Now I need to create coldata information containing the treatmet-control factor
  # and the time covariate
  
  if (time_as_numeric){
    coldata$Time <- coldata$Time / sd(coldata$Time)
  }
  
  return(list(subset_matrix = subset_matrix, coldata = coldata))
}

calculate_auc_trap <- function(values, times) {
  auc <- sum(diff(times) * (head(values, -1) + tail(values, -1)) / 2)
  return(auc)
}


calculate_average_AUC <- function(X, treatment){
  # Identify unique time points by removing replicate suffixes
  time_points <- unique(gsub("_(\\d+)$", "", colnames(X)))
  
  # Average across replicates (for the fitted values, the result replicates have the same value!)
  averaged_data <- sapply(time_points, function(tp) {
    replicate_cols <- grep(paste0("^", tp, "_"), colnames(X), value = TRUE)
    if (length(replicate_cols) == 1) {
      return(as.numeric(X[, replicate_cols]))  
    }
    rowMeans(X[, replicate_cols], na.rm = TRUE)
  })
  
  # Get control samples
  averaged_data_control <- averaged_data[, grep("^control", colnames(averaged_data))]
  # Put the time as column and map to actual hours
  colnames(averaged_data_control) <-  gsub("^control_(\\d+)", "\\1", colnames(averaged_data_control))
  # Get treatment samples
  averaged_data_treatment <- averaged_data[, grep(paste0("^", treatment), colnames(averaged_data))]
  # Put the time as column and map to actual hours
  colnames(averaged_data_treatment) <- gsub(paste0("^", treatment, "_(\\d+)"), "\\1", colnames(averaged_data_treatment))
  
  # Order
  averaged_data_control <- averaged_data_control[, order(as.numeric(colnames(averaged_data_control)))]
  averaged_data_treatment <- averaged_data_treatment[, order(as.numeric(colnames(averaged_data_treatment)))]
  
  auc_results <- data.frame(gene = rownames(X), AUC_Control = NA, AUC_Treatment = NA, Log2_Fold_Change = NA)
  # Calculate AUC
  # Iterate over rows
  for (i in 1:nrow(X)){
    gene_avg_control <- averaged_data_control[i, ]
    gene_avg_treatment <- averaged_data_treatment[i, ]
    
    # Put columns in order:
    auc_control <- calculate_auc_trap(gene_avg_control, as.numeric(names(gene_avg_control)))
    auc_treatment <- calculate_auc_trap(gene_avg_treatment, as.numeric(names(gene_avg_treatment)))
    
    # Calculate log2 fold change
    log2_fc <- log2(auc_treatment / auc_control)
    
    # Store the results
    auc_results$AUC_Control[i] <- auc_control
    auc_results$AUC_Treatment[i] <- auc_treatment
    auc_results$Log2_Fold_Change[i] <- log2_fc
  }
  
  return(auc_results)
  
}


calculate_response_amplitude <-function(X, treatment){
  time_points <- unique(gsub("_(\\d+)$", "", colnames(X)))
  
  # Average across replicates (for the fitted values, the result replicates have the same value!)
  averaged_data <- sapply(time_points, function(tp) {
    replicate_cols <- grep(paste0("^", tp, "_"), colnames(X), value = TRUE)
    if (length(replicate_cols) == 1) {
      return(as.numeric(X[, replicate_cols]))  
    }
    rowMeans(X[, replicate_cols], na.rm = TRUE)
  })
  
  # Get control samples
  averaged_data_control <- averaged_data[, grep("^control", colnames(averaged_data))]
  # Put the time as column and map to actual hours
  colnames(averaged_data_control) <-  gsub("^control_(\\d+)", "\\1", colnames(averaged_data_control))
  # Get treatment samples
  averaged_data_treatment <- averaged_data[, grep(paste0("^", treatment), colnames(averaged_data))]
  
  fold_change <- averaged_data_treatment / averaged_data_control
  fold_change <- log2(fold_change)
  out <- do.call(rbind, apply(fold_change, 1, function(x) x[which.max(abs(x))]))
  colnames(out) <- c("amplitude")
  out <- cbind(gene = rownames(out), out)
  rownames(out) <- NULL
  out <- data.frame(out)
  out$amplitude <-  as.numeric(out$amplitude)
  return(out)
}

DE_analysis <- function(X, treatments, store_folder= ".", name = "."){
  final_results <- data.frame()
  for (treat in treatments) {  # all w.r.t control
    print(treat)
    Z <- create_subset_matrix(X, treat)
    print(Z)
    design_formula <- as.formula("~ Treatment * Time")
    ddsMat <- DESeqDataSetFromMatrix(countData = Z$subset_matrix,
                                     colData = Z$coldata,
                                     design = design_formula)
    
    reduced_design_formula = as.formula("~ Time")
    # Perform differential expression analysis with LRT
    dds <- DESeq(ddsMat, test = "LRT", reduced = reduced_design_formula)
    # Extract design matrix and fitted coefficients
    design_matrix <- model.matrix(design_formula, data = Z$coldata) 
    # Store the design matrix
    write.csv(design_matrix, file = paste0(store_folder, "/design_matrix_", name, "_", treat, ".csv"), row.names = TRUE)
    betas <- coef(dds)
    # Store betas 
    write.csv(betas, file = paste0(store_folder, "/betas_", name, "_", treat, ".csv"), row.names = TRUE)
    fitted_log2 <- design_matrix %*% t(betas)
    fitted_values <- t(2^fitted_log2) # These are NORMALIZED VALUES!
    # Store normalized fitted values
    write.csv(fitted_values, file = paste0(store_folder, "/fitted_values_", name, "_", treat, ".csv"), row.names = TRUE)
    
    average_normalized_values <- data.frame(rowMeans(fitted_values))
    average_normalized_values <- cbind('gene' = rownames(average_normalized_values), average_normalized_values)
    colnames(average_normalized_values) <- c("gene", "average")
    
    auc_results <- calculate_average_AUC(fitted_values, treat)
    auc_results <- auc_results[, c("gene", "Log2_Fold_Change")]
    amplitude_results <- calculate_response_amplitude(fitted_values, treat)
    # Extract results
    res <- results(dds)
    # Extract gene names and padj (adjusted p-values)
    # Assuming rownames(res) contains gene names
    res_df <- as.data.frame(res)
    res_df$gene <- rownames(res_df)  # Add gene names as a column
    res_df$treatment <- treat  # Add treatment as a column
    
    # Select relevant columns (gene name, padj, and treatment)
    res_selected <- res_df[, c("gene", "padj", "stat", "treatment")]
    res_selected <- merge(res_selected, auc_results, by = "gene", all =TRUE)
    res_selected <- merge(res_selected, amplitude_results,  by = "gene", all = TRUE)
    res_selected <- merge(res_selected, average_normalized_values, by ='gene')
    
    # Append to the final results data frame
    final_results <- rbind(final_results, res_selected)
  }
  
  final_results$neg_log10_pvalue <- -log10(final_results$padj)
  
  return(final_results)
}


plot_volcano <- function(X, treat, log_2_FC_threshold = 1) {
  # Remove NA
  X <- X[X$treatment == treat, ]
  X$Log2_Fold_Change[is.infinite(X$Log2_Fold_Change)] <- NaN
  X$neg_log10_pvalue[is.infinite(X$neg_log10_pvalue)] <- NaN
  X <- X[complete.cases(X), ]
  
  
  # Assign color categories
  X$color <- ifelse(
    X$Log2_Fold_Change <= -log_2_FC_threshold, "blue",
    ifelse(X$Log2_Fold_Change >= log_2_FC_threshold, "red", "black")
  )
  
  # Create the plot
  return(ggplot(X) +
           # Add histogram as background
           geom_histogram(aes(x = Log2_Fold_Change, y = ..count../max(..count..) * max(X$neg_log10_pvalue)),
                          bins = 30, fill = "gray", alpha = 0.3) +
           # Add scatter plot
           geom_point(aes(x = Log2_Fold_Change, y = neg_log10_pvalue, color = color), alpha = 0.3) +
           # Custom colors for points
           scale_color_manual(values = c("blue", "black", "red")) +
           # Labels
           labs(title = treat,
                x = "Log2 Fold Change",
                y = "-Log10 P-value") +
           # Add secondary y-axis for histogram counts
           scale_y_continuous(
             sec.axis = sec_axis(~ . / max(X$neg_log10_pvalue) * max(table(cut(X$Log2_Fold_Change, 30))),
                                 name = "Histogram Counts")) +
           # Set x-axis limits
           xlim(-10, 10) +
           theme_minimal() +
           theme(legend.position = "none"))
}



