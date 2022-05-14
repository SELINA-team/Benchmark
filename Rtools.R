suppressMessages(library(stringr))
suppressMessages(library(data.table))
suppressMessages(library(Seurat))
suppressMessages(library(scibet))
suppressMessages(library(SingleCellExperiment))
suppressMessages(library(scmap))
suppressMessages(library(SingleR))
suppressMessages(library(scran))
suppressMessages(library(CelliD))
suppressMessages(library(scPred))
suppressMessages(library(CHETAH))
suppressMessages(library(singleCellNet))
suppressMessages(library(sva))
suppressMessages(library(Rcpp))
suppressMessages(library(dplyr))
suppressMessages(library(purrr))
suppressMessages(library(SummarizedExperiment))

tissue <- "Lung"
path <- "/fs/home/renpengfei/performance/"
path_in <- paste0(path,tissue,'/data/')
path_out <- paste0(path,tissue,'/res/')

samples <- list.files(paste0(path,tissue,'/rds/'), pattern = "rds$", recursive = F)
samples <- gsub(".rds", "", samples)

res <- list()

# evaluate
evaluate <- function(true_lab, pred_res) {
  "
    Conf: confusion matrix
    macro : macro F1-score
    F1 : F1-score per class
    Acc : accuracy
    PercUnl : percentage of unlabeled cells
    PopSize : number of cells per cell type
    "
  pred_lab <- pred_res[[1]]
  time <- pred_res[[2]][1]
  memory <- pred_res[[2]][2]
  unique_true <- unlist(unique(true_lab))
  unique_pred <- unlist(unique(pred_lab))

  unique_all <- unique(c(unique_true, unique_pred))
  conf <- table(true_lab, pred_lab)
  pop_size <- rowSums(conf)

  conf_F1 <- table(true_lab, pred_lab)
  F1 <- vector()
  sum_acc <- 0

  for (i in c(1:length(unique_true))) {
    findLabel <- colnames(conf_F1) == row.names(conf_F1)[i]
    if (sum(findLabel)) {
      prec <- conf_F1[i, findLabel] / colSums(conf_F1)[findLabel]
      rec <- conf_F1[i, findLabel] / rowSums(conf_F1)[i]
      if (prec == 0 || rec == 0) {
        F1[i] <- 0
      } else {
        F1[i] <- (2 * prec * rec) / (prec + rec)
      }
      sum_acc <- sum_acc + conf_F1[i, findLabel]
    } else {
      F1[i] <- 0
    }
  }

  pop_size <- pop_size[pop_size > 0]
  names(F1) <- names(pop_size)
  macro_F1 <- mean(F1)
  total <- length(pred_lab)
  num_unlab <- sum(pred_lab == "unassigned") + sum(pred_lab == "Unassigned") + sum(pred_lab == "rand") + sum(pred_lab == "Unknown") + sum(pred_lab == "unknown") + sum(pred_lab == "Node") + sum(pred_lab == "ambiguous") + sum(pred_lab == "unassign")
  per_unlab <- num_unlab / total
  acc <- sum_acc / sum(conf_F1)

  result <- list(Acc = acc, Macro_F1 = macro_F1, Time = time, Memory = memory, Conf = conf, F1 = F1)
  return(result)
}

scibet_pre <- function(ref_expr, query_expr) {
    tryCatch(
        expr = {
            Rprof("./TM_log/TM_scibet.txt", memory.profiling = TRUE, append = FALSE)
            true_lab <- colnames(query_expr)
            labels <- colnames(ref_expr)
            ref_expr <- as.data.frame(t(ref_expr))
            ref_expr$label <- labels
            query_expr <- t(query_expr)
            etest_gene <- SelectGene(ref_expr, k = 1000)
            prd <- SciBet(ref_expr, query_expr)
            Rprof(NULL)
            TM_profile <- summaryRprof("./TM_log/TM_scibet.txt", memory = "both")
            time <- TM_profile[["by.total"]]['"scibet_pre"', "total.time"]
            memory <- TM_profile[["by.total"]]['"scibet_pre"', "mem.total"]
            pre.res <- list(prediction = prd, TMprofile = c(time = time, memory = memory))
            res <- evaluate(true_lab, pre.res)
            print(paste0('scibet: ',res[c(1,2,3)]))
            return(res)
        },
        error = function(e){
            message('Caught an error!')
            print(paste0('Error: ',e))
        }
    )
}

scmap_cell_pre <- function(ref_expr, query_expr) {
    tryCatch(
        expr = {
            Rprof("./TM_log/TM_scmap_cell.txt", memory.profiling = TRUE, append = FALSE)
            true_lab <- colnames(query_expr)
            celltype <- as.data.frame(as.factor(colnames(ref_expr)))
            colnames(celltype) <- "celltype1"
            ref_expr <- SingleCellExperiment(assays = list(normcounts = as.matrix(ref_expr)), colData = celltype)
            logcounts(ref_expr) <- log2(normcounts(ref_expr) + 1)
            rowData(ref_expr)$feature_symbol <- rownames(ref_expr)
            ref_expr <- ref_expr[!duplicated(rownames(ref_expr)), ]
            ref_expr <- selectFeatures(ref_expr, suppress_plot = FALSE)
            ref_expr <- indexCell(ref_expr)

            query_expr <- SingleCellExperiment(assays = list(normcounts = as.matrix(query_expr)))
            logcounts(query_expr) <- log2(normcounts(query_expr) + 1)
            rowData(query_expr)$feature_symbol <- rownames(query_expr)

            scmap_pre <- scmapCell(
                query_expr,
                list(
                  ref_expr@metadata$scmap_cell_index
                )
            )
            cells <- as.data.frame(scmap_pre[[1]][[1]])
            scores <- as.data.frame(scmap_pre[[1]][[2]])
            cells <- t(cells)
            scores <- t(scores)
            max_scores <- max.col(scores)
            pre_cells <- c()
            for (i in 1:nrow(cells)) {
                pre_cells[i] <- cells[i, max_scores[i]]
            }
            pre_cells <- as.numeric(pre_cells)
            scmap_results <- list()
            for (j in 1:length(pre_cells)) {
                scmap_results[[j]] <- celltype[pre_cells[j], 1]
            }
            scmap_results <- as.character(unlist(scmap_results))
            Rprof(NULL)
            TM_profile <- summaryRprof("./TM_log/TM_scmap_cell.txt", memory = "both")
            time <- TM_profile[["by.total"]]['"scmap_cell_pre"', "total.time"]
            memory <- TM_profile[["by.total"]]['"scmap_cell_pre"', "mem.total"]
            pre.res <- list(prediction = scmap_results, TMprofile = c(time = time, memory = memory))
            res <- evaluate(true_lab, pre.res)
            print(paste0('scmap_cell: ',res[c(1,2,3)]))
            return(res)
        },
        error = function(e){
            message('Caught an error!')
            print(paste0('Error: ',e))
        }
    )
}

singleR_pre <- function(ref_expr, query_expr) {
    tryCatch(
        expr = {
            Rprof("./TM_log/TM_singleR.txt", memory.profiling = TRUE, append = FALSE)
            true_lab <- colnames(query_expr)
            ref_expr <- SummarizedExperiment(assays=list(counts=ref_expr))
            query_expr <- SummarizedExperiment(assays=list(counts=query_expr))
			      ref_expr <- logNormCounts(ref_expr)
            query_expr <- logNormCounts(query_expr)
            pred.grun <- SingleR(test = query_expr, ref = ref_expr, labels = colnames(ref_expr), de.method = "wilcox")
            Rprof(NULL)
            TM_profile <- summaryRprof("./TM_log/TM_singleR.txt", memory = "both")
            time <- TM_profile[["by.total"]]['"singleR_pre"', "total.time"]
            memory <- TM_profile[["by.total"]]['"singleR_pre"', "mem.total"]
            pre.res <- list(prediction = pred.grun$labels, TMprofile = c(time = time, memory = memory))
            res <- evaluate(true_lab, pre.res)
            print(paste0('singleR: ',res[c(1,2,3)]))
            return(res)
        },
        error = function(e){
            message('Caught an error!')
            print(paste0('Error: ',e))
        } 
    )
}

cellid_pre <- function(ref_expr, query_expr) {
    tryCatch(
        expr = {
            Rprof("./TM_log/TM_cellid.txt", memory.profiling = TRUE, append = FALSE)
            true_lab <- colnames(query_expr)
            ref_label <- colnames(ref_expr)
            ref_expr <- CreateSeuratObject(counts = ref_expr)
            query_expr <- CreateSeuratObject(counts = query_expr)
            ref_expr <- NormalizeData(ref_expr)
            ref_expr <- ScaleData(ref_expr, features = rownames(ref_expr))
            ref_expr$cell.type <- ref_label
            ref_expr <- RunMCA(ref_expr)
            ref_gs <- GetCellGeneSet(ref_expr, dims = 1:50, n.features = 200)

            query_expr <- NormalizeData(query_expr)
            query_expr <- FindVariableFeatures(query_expr)
            query_expr <- ScaleData(query_expr)
            query_expr <- RunMCA(query_expr, nmcs = 50)

            HGT_ref_gs <- RunCellHGT(query_expr, pathways = ref_gs, dims = 1:50)
            ref_gs_match <- rownames(HGT_ref_gs)[apply(HGT_ref_gs, 2, which.max)]
            ref_gs_prediction <- ref_expr$cell.type[ref_gs_match]
            ref_gs_prediction_signif <- ifelse(apply(HGT_ref_gs, 2, max) > 2, yes = ref_gs_prediction, "unassigned")
            ref_gs_prediction <- ref_gs_prediction_signif
            Rprof(NULL)
            TM_profile <- summaryRprof("./TM_log/TM_cellid.txt", memory = "both")
            time <- TM_profile[["by.total"]]['"cellid_pre"', "total.time"]
            memory <- TM_profile[["by.total"]]['"cellid_pre"', "mem.total"]
            pre.res <- list(prediction = ref_gs_prediction, TMprofile = c(time = time, memory = memory))
            res <- evaluate(true_lab, pre.res)
            print(paste0('cellid: ',res[c(1,2,3)]))
            return(res)
        },
        error = function(e){
            message('Caught an error!')
            print(paste0('Error: ',e))
        }
    )
}

count2seurat <- function(counts) {
  cells <- ncol(counts)
  if (cells <= 100000) {
    npc <- 50
  } else {
    npc <- 100
  }
  data <- CreateSeuratObject(counts = counts)
  data <- NormalizeData(data, verbose = FALSE)
  data <- FindVariableFeatures(data, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
  data <- ScaleData(data, verbose = FALSE)
  data <- RunPCA(data, verbose = FALSE, features = VariableFeatures(data), npcs = npc)
  pc.contribution <- data@reductions$pca@stdev / sum(data@reductions$pca@stdev) * 100
  pc.contribution.cum <- cumsum(pc.contribution)
  pc.first <- which(pc.contribution.cum > 75)[1]
  dims.use <- 1:pc.first
  data <- RunUMAP(object = data, reduction = "pca", dims = dims.use)
  return(data)
}

# scPred_pre <- function(ref_expr, query_expr) {
#     tryCatch(
#         expr = {
#             Rprof("./TM_log/TM_scPred.txt", memory.profiling = TRUE, append = FALSE)
#             true_lab <- colnames(query_expr)
#             label <- colnames(ref_expr)
#             ref_expr <- count2seurat(ref_expr)
#             ref_expr$cell_type <- label
#             ref_expr <- getFeatureSpace(ref_expr, "cell_type")
#             ref_expr <- trainModel(ref_expr)
#             query_expr <- count2seurat(query_expr)
#             query_expr <- scPredict(query_expr, ref_expr)
#             Rprof(NULL)
#             TM_profile <- summaryRprof("./TM_log/TM_scPred.txt", memory = "both")
#             time <- TM_profile[["by.total"]]['"scPred_pre"', "total.time"]
#             memory <- TM_profile[["by.total"]]['"scPred_pre"', "mem.total"]
#             pre.res <- list(prediction = query_expr$scpred_prediction, TMprofile = c(time = time, memory = memory))
#             res <- evaluate(true_lab, pre.res)
#             print(paste0('scPred: ',res[c(1,2,3)]))
#             return(res)
#         },
#         error = function(e){
#             message('Caught an error!')
#             print(paste0('Error: ',e))
#         }
#     ) 
# }

# CHETAH_pre <- function(ref_expr, query_expr) {
#     tryCatch(
#         expr = {
#             Rprof("./TM_log/TM_CHETAH.txt", memory.profiling = TRUE, append = FALSE)
#             true_lab <- colnames(query_expr)
#             ref_expr <- SingleCellExperiment(
#                 assays = list(counts = ref_expr),
#                 colData = DataFrame(celltypes = colnames(ref_expr))
#             )
#             query_seurat <- count2seurat(query_expr)
#             query_expr <- SingleCellExperiment(
#                 assays = list(counts = test_mat),
#                 reducedDims = SimpleList(TSNE = query_seurat@reductions$umap@cell.embeddings)
#             )
#             query_expr <- CHETAHclassifier(input = query_expr, ref_cells = ref_expr)
#             Rprof(NULL)
#             TM_profile <- summaryRprof("./TM_log/TM_CHETAH.txt", memory = "both")
#             time <- TM_profile[["by.total"]]['"CHETAH_pre"', "total.time"]
#             memory <- TM_profile[["by.total"]]['"CHETAH_pre"', "mem.total"]
#             pre.res <- list(prediction = query_expr$celltype_CHETAH, TMprofile = c(time = time, memory = memory))
#             res <- evaluate(true_lab, pre.res)
#             print(paste0('CHETAH: ',res[c(1,2,3)]))
#             return(res)
#         },
#         error = function(e){
#             message('Caught an error!')
#             print(paste0('Error: ',e))
#         }
#     )
# }

scn_pre <- function(ref_expr, query_expr) {
    tryCatch(
        expr = {
            Rprof("./TM_log/TM_scn.txt", memory.profiling = TRUE, append = FALSE)
            true_lab <- colnames(query_expr)
            label <- colnames(ref_expr)
            ref_cell_id <- as.character(1:ncol(ref_expr))
            colnames(ref_expr) <- ref_cell_id
            meta <- data.frame(label = label, cell_id = ref_cell_id)
            class_info <- scn_train(stTrain = meta, expTrain = ref_expr, dLevel = "label", colName_samp = "cell_id")
            res <- scn_predict(cnProc = class_info[["cnProc"]], expDat = query_expr)
            label <- rownames(res)
            pred <- c()
            for (i in 1:(ncol(res) - 2)) {
                pred[i] <- label[which.max(res[, i])]
            }
            Rprof(NULL)
            TM_profile <- summaryRprof("./TM_log/TM_scn.txt", memory = "both")
            time <- TM_profile[["by.total"]]['"scn_pre"', "total.time"]
            memory <- TM_profile[["by.total"]]['"scn_pre"', "mem.total"]
            pre.res <- list(prediction = pred, TMprofile = c(time = time, memory = memory))
            res <- evaluate(true_lab, pre.res)
            print(paste0('scn: ',res[c(1,2,3)]))
            return(res)
        },
        error = function(e){
            message('Caught an error!')
            print(paste0('Error: ',e))
        }
    )
}

for (sample in samples) {

    print(sample)
    train_mat <- fread(paste0(path_in, sample, "_train.txt"), sep = "\t", nThread = 6, header = TRUE)
    test_mat <- fread(paste0(path_in, sample, "_test.txt"), sep = "\t", nThread = 6, header = TRUE)
    genes <- read.table(paste0(path_in, sample, "_gene.txt"), sep = "\t", header = FALSE)
    train_mat <- as.matrix(train_mat)
    test_mat <- as.matrix(test_mat)
    rownames(train_mat) <- rownames(test_mat) <- genes[,1]

    colnames(test_mat) <- unlist(lapply(colnames(test_mat), function(x) {
        x <- unlist(str_split(x, ";"))[1]
    }))
    colnames(train_mat) <- unlist(lapply(colnames(train_mat), function(x) {
        x <- unlist(str_split(x, ";"))[1]
    }))   


    print("[info]:scmap_cell")
    scmap_cell.res <- scmap_cell_pre(train_mat, test_mat)

    
    print("[info]:singleR")
    singleR.res <- singleR_pre(train_mat, test_mat)


    # print("[info]:CHETAH")
    # CHETAH.res <- CHETAH_pre(train_mat, test_mat)

    
    print("[info]:scn")
    scn.res <- scn_pre(train_mat, test_mat)


    # print("[info]:scPred")
    # scPred.res <- scPred_pre(train_mat, test_mat)


    print("[info]:scibet")
    scibet.res <- scibet_pre(train_mat, test_mat)


    print("[info]:cellid")
    cellid.res <- cellid_pre(train_mat, test_mat)


    res[[sample]] <- list(scmap = scmap_cell.res, SingleR = singleR.res, SingleCellNet = scn.res, scibet = scibet.res, CelliD = cellid.res)
}

saveRDS(res, file = paste0(path_out,tissue,"_Rtools.rds"))
