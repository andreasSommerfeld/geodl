# from: https://wvview.org/gslr/c12CNN.html
#
# chapter 12

# library -----------------------------------------------------------------

.libPaths( "C:/R/R-4.5.2/library")

library(tidyverse)
library(terra)
library(torch)
library(torchvision)
library(luz)
library(yardstick)
library(micer)
library(gt)


# set up torch via locally downloaded binaries ----------------------------

Sys.setenv(TORCH_URL= "C:/temp/libtorch-win-shared-with-deps-2.7.1+cpu.zip")
Sys.setenv(LANTERN_URL="C:/temp/lantern-0.16.3+cpu-win64.zip")

torch::install_torch()


# test skript  ------------------------------------------------------------

outpath <- "E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/cnn_test_data/outpath/"
cnn_logs <- "E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/cnn_test_data/outpath/cnnLogs.csv"

fldPth <- "E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/cnn_test_data/archive/EuroSATallBands/"

trainDF <- read_csv(str_glue("{fldPth}train.csv"))
valDF <- read_csv(str_glue("{fldPth}validation.csv"))
testDF <- read_csv(str_glue("{fldPth}test.csv"))

eurosatDataSet <- torch::dataset(

  name = "eurosatDataSet",

  initialize = function(df,
                        pth,
                        bands,
                        doAugs=FALSE){
    self$df <- df
    self$pth <- pth
    self$bands <- bands
    self$doAugs <- doAugs
  },

  .getitem = function(i){

    imgName <- unlist(self$df[i, "Filename"], use.names=FALSE)

    label <- unlist(self$df[i, "Label"], use.names=FALSE) |>
      torch_tensor(dtype=torch_int64()) |>
      torch_add(1)
    label <- label$squeeze()
    label <- label |> torch_tensor(dtype=torch_int64())

    img <- rast(paste0(self$pth, imgName)) |>
      terra::subset(self$bands) |>
      as.array() |>
      torch_tensor(dtype=torch_float32())

    img <- img$permute(c(3,2,1))

    img <- img/10000

    if(self$doAugs == TRUE){
      img <- torchvision::transform_random_horizontal_flip(img, p=0.5)

      img <- torchvision::transform_random_vertical_flip(img, p=0.5)
    }

    return(list(preds = img, label = label))
  },

  .length = function(){

    return(nrow(self$df))

  }
)

trainDS <- eurosatDataSet(df=trainDF,
                          pth=fldPth,
                          bands=c(2,3,4,5,6,7,8,9,12,13),
                          doAugs=TRUE)

valDS <- eurosatDataSet(df=valDF,
                        pth=fldPth,
                        bands=c(2,3,4,5,6,7,8,9,12,13),
                        doAugs=FALSE)

testDS <- eurosatDataSet(df=testDF,
                         pth=fldPth,
                         bands=c(2,3,4,5,6,7,8,9,12,13),
                         doAugs=FALSE)

trainDL <- torch::dataloader(trainDS,
                             batch_size=32,
                             shuffle=TRUE,
                             drop_last = TRUE)

valDL <- torch::dataloader(valDS,
                           batch_size=32,
                           shuffle=FALSE,
                           drop_last = TRUE)

testDL <- torch::dataloader(testDS,
                            batch_size=32,
                            shuffle=FALSE,
                            drop_last = FALSE)

batch1 <- trainDL$.iter()$.next()

batch1$preds$shape
torch_mean(batch1$preds, dim=c(1,3,4))



# cnn from scratch --------------------------------------------------------

myCNN <- nn_module(
  "convnet",

  initialize = function(inChn=3,
                        nFMs=c(32, 64, 128, 256),
                        nNodes=c(512, 256),
                        nCls=3) {

    self$inChn = inChn
    self$nFMs = nFMs
    self$nNodes = nNodes
    self$nCls = nCls

    self$cnnComp <- nn_sequential(
      nn_conv2d(inChn, nFMs[1], kernel_size=3, padding=1),
      nn_batch_norm2d(nFMs[1]),
      nn_relu(),
      nn_max_pool2d(kernel_size=2),
      nn_conv2d(nFMs[1], nFMs[2], kernel_size=3, padding=1),
      nn_batch_norm2d(nFMs[2]),
      nn_relu(),
      nn_max_pool2d(kernel_size=2),
      nn_conv2d(nFMs[2], nFMs[3], kernel_size=3, padding=1),
      nn_batch_norm2d(nFMs[3]),
      nn_relu(),
      nn_max_pool2d(kernel_size=2),
      nn_conv2d(nFMs[3], nFMs[4], kernel_size=3, padding=1),
      nn_batch_norm2d(nFMs[4]),
      nn_relu()
    )

    self$fcComp <- nn_sequential(
      nn_linear(nFMs[4]*8*8, nNodes[1]),
      nn_batch_norm1d(nNodes[1]),
      nn_relu(),
      nn_linear(nNodes[1], nNodes[2]),
      nn_batch_norm1d(nNodes[2]),
      nn_relu(),
      nn_linear(nNodes[2], nCls),
    )

  },

  forward = function(x) {

    x <- self$cnnComp(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$fcComp(x)
    return(x)

  }
)

testT <- torch_rand(32, 10, 64, 64)
testTPred <- model(testT)

testTPred$shape

count_trainable_params <- function(model) {
  if (!inherits(model, "nn_module")) {
    stop("The input must be a torch nn_module.")
  }

  params <- model$parameters

  trainable_params <- lapply(params, function(param) {
    if (param$requires_grad) {
      as.numeric(prod(param$size()))
    } else {
      0
    }
  })

  total_trainable_params <- sum(unlist(trainable_params))

  return(total_trainable_params)
}

count_trainable_params(model)

fitted <- myCNN |>
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics=luz_metric_accuracy()
  ) |>
  set_hparams(
    inChn=10,
    nFMs=c(32, 64, 128, 256),
    nNodes=c(512, 256),
    nCls=10
  ) |>
  fit(data = trainDL,
      epochs = 25,
      valid_data = valDL,
      callbacks = list(luz_callback_csv_logger("E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/cnn_test_data/outpath/cnnLogs.csv"),
                       luz_callback_keep_best_model(monitor = "valid_loss",
                                                    mode = "min",
                                                    min_delta = 0)),
      accelerator = accelerator(device_placement = TRUE,
                                cpu = FALSE,
                                cuda_index = torch::cuda_current_device()),
      verbose=TRUE)

luz_save(fitted, paste0(outpath, "/cnnModel.pt"))
