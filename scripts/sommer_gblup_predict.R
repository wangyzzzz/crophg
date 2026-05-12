args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript scripts/sommer_gblup_predict.R <fit_data.csv> <grm.csv> <predictions.csv>")
}

fit_csv <- args[[1]]
grm_csv <- args[[2]]
out_csv <- args[[3]]

suppressPackageStartupMessages({
  library(sommer)
})

fit_df <- read.csv(fit_csv, stringsAsFactors = FALSE, check.names = FALSE)
grm_df <- read.csv(grm_csv, stringsAsFactors = FALSE, check.names = FALSE)

required_fit_cols <- c("plot_id", "genotype_id", "y")
missing_fit_cols <- setdiff(required_fit_cols, names(fit_df))
if (length(missing_fit_cols) > 0) {
  stop(paste0("fit_data.csv 缺少列: ", paste(missing_fit_cols, collapse = ", ")))
}
if (!("genotype_id" %in% names(grm_df))) {
  stop("grm.csv 缺少 genotype_id 列。")
}

rownames(grm_df) <- as.character(grm_df$genotype_id)
grm_df$genotype_id <- NULL
grm_mat <- as.matrix(grm_df)
storage.mode(grm_mat) <- "double"

fit_df$plot_id <- as.character(fit_df$plot_id)
fit_df$genotype_id <- as.character(fit_df$genotype_id)
fit_df$geno_factor <- factor(fit_df$genotype_id, levels = rownames(grm_mat))
fit_df$y <- as.numeric(fit_df$y)

missing_grm <- unique(fit_df$genotype_id[is.na(fit_df$geno_factor)])
if (length(missing_grm) > 0) {
  stop(paste0("以下 genotype_id 不在 GRM 中: ", paste(head(missing_grm, 10), collapse = ", ")))
}

train_df <- fit_df[is.finite(fit_df$y), , drop = FALSE]
pred_df <- fit_df[, c("plot_id", "genotype_id"), drop = FALSE]
pred_df$y <- NA_real_
pred_df$geno_factor <- factor(pred_df$genotype_id, levels = levels(train_df$geno_factor))

ans <- mmer(
  fixed = y ~ 1,
  random = ~ vsr(geno_factor, Gu = grm_mat),
  rcov = ~ units,
  data = train_df,
  verbose = FALSE
)

pred_out <- tryCatch(
  {
    predict(ans, newdata = pred_df)
  },
  error = function(e) {
    NULL
  }
)

pred_values <- NULL
if (!is.null(pred_out)) {
  if (is.list(pred_out) && "pvals" %in% names(pred_out)) {
    pvals <- as.data.frame(pred_out$pvals)
    if ("predicted.value" %in% names(pvals)) {
      pred_values <- pvals$predicted.value
    } else {
      numeric_cols <- names(pvals)[vapply(pvals, is.numeric, logical(1))]
      if (length(numeric_cols) > 0) {
        pred_values <- pvals[[numeric_cols[1]]]
      }
    }
  } else if (is.data.frame(pred_out) && "predicted.value" %in% names(pred_out)) {
    pred_values <- pred_out$predicted.value
  }
}

if (is.null(pred_values)) {
  u_name <- names(ans$U)[1]
  if (is.null(u_name) || nchar(u_name) == 0) {
    stop("sommer 输出中未找到随机效应 U。")
  }
  u_df <- as.data.frame(ans$U[[u_name]])
  u_cols <- names(u_df)
  geno_col <- if ("geno_factor" %in% u_cols) "geno_factor" else if ("Name" %in% u_cols) "Name" else NA_character_
  u_col <- if ("u:geno_factor" %in% u_cols) "u:geno_factor" else if ("u" %in% u_cols) "u" else NA_character_

  if (is.na(geno_col)) {
    u_df$genotype_id <- rownames(u_df)
  } else {
    u_df$genotype_id <- as.character(u_df[[geno_col]])
  }
  if (is.na(u_col)) {
    numeric_cols <- u_cols[vapply(u_df[u_cols], is.numeric, logical(1))]
    if (length(numeric_cols) == 0) {
      stop("无法识别 sommer U 输出中的数值随机效应列。")
    }
    u_col <- numeric_cols[1]
  }

  beta_obj <- ans$Beta
  beta_numeric <- suppressWarnings(as.numeric(beta_obj))
  intercept <- beta_numeric[is.finite(beta_numeric)][1]
  if (!is.finite(intercept)) {
    intercept <- 0.0
  }

  pred_map <- intercept + as.numeric(u_df[[u_col]])
  names(pred_map) <- as.character(u_df$genotype_id)
  pred_values <- pred_map[pred_df$genotype_id]
}

pred_values <- as.numeric(pred_values)
if (length(pred_values) != nrow(pred_df)) {
  stop("sommer 预测输出长度与输入样本数不一致。")
}

out_df <- data.frame(
  genotype_id = pred_df$genotype_id,
  predicted.value = pred_values,
  stringsAsFactors = FALSE
)

missing_pred <- unique(out_df$genotype_id[!is.finite(out_df$predicted.value)])
if (length(missing_pred) > 0) {
  stop(paste0("以下 genotype_id 未得到 GBLUP 预测值: ", paste(head(missing_pred, 10), collapse = ", ")))
}

out_df <- out_df[!duplicated(out_df$genotype_id), , drop = FALSE]
write.csv(out_df, out_csv, row.names = FALSE)
