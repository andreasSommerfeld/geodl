
# source: https://wvview.org/gslr/C15geodl_P1.html chapter 15

# library -----------------------------------------------------------------

.libPaths( "C:/R/R-4.5.2/library")

library(tidyverse)
library(terra)
library(sf)
library(tmap)
library(tmaptools)
library(gt)
library(geodl)

indata <- "E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/"
outdata <- "E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/outdata/"


fldPth <- indata
outPth <- outdata

img <- rast(str_glue("{fldPth}images/test_clip_raster_4band.tif"))
ext <- st_read(str_glue("{fldPth}extent/extent.shp"), quiet=TRUE)
conifers <- st_read(str_glue("{fldPth}mask_vector/land_cover_type.shp"), quiet=TRUE) %>%
  filter(lc_name == "confi")
lc <- st_read(str_glue("{fldPth}mask_vector/land_cover_type.shp"), quiet=TRUE)



# -------------------------------------------------------------------------

image_folder <- "E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/"

mask_folder <- paste0(image_folder, "masks/")

files <- list.files(image_folder,
           pattern="*.jp2")

ext <- st_read("E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/extent/tree_spec_stand_vec.shp") %>%
  mutate(id_img = seq(0,72,1))

# loop to create masks for input images

output <- vector("list", length(files))

for (i in seq_along(output)) {

print(i)

file_name <- files[i]

image_id <- as.numeric(strsplit( (strsplit(file_name, "[_]")[[1]][5]), "[.]" )[[1]][1])

ext_loop <- ext %>%         # choose extent
  filter(id_img == image_id)

plot(st_geometry(ext_loop)) # plot extent

makeMasks(image = paste0(image_folder, file_name),
          features = ext_loop,
          crop = TRUE,
          extent = ext_loop,
          field = "classvalue",
          background = 0,
          outMask = paste0(mask_folder, strsplit(file_name, "[.]")[[1]][1], "_mask.tif"),
          mode = "Mask")

}


dir.create(paste0(image_folder, "chips"))

for (i in seq_along(output)) {

  print(i)

  file_name <- files[i]

  image_id <- as.numeric(strsplit( (strsplit(file_name, "[_]")[[1]][5]), "[.]" )[[1]][1])
  image_name <- strsplit( file_name, "[.]" )[[1]][1]

  makeChipsMultiClass(image = paste0(image_folder, file_name),
                      mask =  paste0(image_folder, "masks/", image_name, "_mask.tif"),
                      n_channels = 4,
                      #change to 192 (dividable by 16) below
                      size = 200, # rational for 200 pixel * 0,2m / pixel = 40m at least one old growth tree crown is could be within image, even smallest area can be used (size        : 225, 362, 5  (nrow, ncol, nlyr))
                      stride_x = 100,
                      stride_y = 100,
                      outDir = paste0(image_folder, "chips"),
                      useExistingDir=FALSE)



}



tree_spec_chips_df <- makeChipsDF(folder = paste0(image_folder, "chips/"),
                         extension=".tif",
                         mode="All",
                         saveCSV=FALSE)

head(tree_spec_chips_df) |>
  gt()


viewChips(chpDF=tree_spec_chips_df,
          folder=paste0(image_folder, "chips/"),
          nSamps = 10,
          mode = "both",
          justPositive = FALSE,
          cCnt = 5,
          rCnt = 2,
          r = 1,
          g = 2,
          b = 3,
          rescale = FALSE,
          rescaleVal = 1,
          cCodes = c(0,1,2),
          cNames= c("Background", "Conifers", "Broardleaves"),
          cColors= c("gray", "red", "green"),
          useSeed = FALSE,
          seed = 42)

tree_spec_chips_ds <- defineSegDataSet(chpDF= tree_spec_chips_df,
                         folder=paste0(image_folder, "chips/"),
                         normalize = FALSE,
                         rescaleFactor = 255,
                         mskRescale = 1,
                         mskAdd = 1, # add 1 to all values
                         bands = c(1, 2, 3, 4), # use r, g, b & nir
                         bMns = 1,
                         bSDs = 1,
                         doAugs = TRUE,
                         maxAugs = 3,
                         probVFlip = .3,
                         probHFlip = .3,
                         probBrightness = .1,
                         probContrast = .1,
                         probGamma = 0,
                         probHue = 0,
                         probRotation = .3,
                         probSaturation = .2,
                         brightFactor = c(0.8, 1.2),
                         contrastFactor = c(0.8, 1.2),
                         gammaFactor = c(0.8, 1.2, 1),
                         hueFactor = c(-0.2, 0.2),
                         saturationFactor = c(0.8, 1.2)
)

tree_spec_chips_dl <- torch::dataloader(tree_spec_chips_ds,
                          batch_size=5,
                          shuffle=TRUE,
                          drop_last = TRUE)

length(tree_spec_chips_dl)

viewBatch(tree_spec_chips_dl,
          nCols = 3,
          r = 1,
          g = 2,
          b = 3,
          cCodes = c(1,2,3),
          cNames= c("Background", "Conifers", "Broardleaves"),
          cColors= c("gray", "red", "green")
)

unetMod <- defineUNet(inChn = 4,
                      nCls = 7,
                      actFunc = "lrelu",
                      useAttn = TRUE,
                      useSE = TRUE,
                      useRes = TRUE,
                      useASPP = TRUE,
                      useDS = FALSE,
                      enChn = c(16, 32, 64, 128),
                      dcChn = c(128, 64, 32, 16),
                      btnChn = 256,
                      dilRates = c(6, 12, 18),
                      dilChn = c(256, 256, 256, 256),
                      negative_slope = 0.01,
                      seRatio = 8
)$to(device="cpu")

t1 <- torch::torch_rand(c(12,4,256,256))$to(device="cpu")

p1 <- unetMod(t1)

p1$shape

countParams(unetMod, t1)

################################################################################
###                               the end                                    ###
################################################################################

# now chapter 16 https://wvview.org/gslr/c16geodl_P2.html -----------------

### create train, validation and test sub data sets
### only to be computed once
#
# ### training images
# files_chips <- list.files("E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/chips/images/",
#                           full.names = TRUE)
#
# {set.seed(88); files_chips_samples <- sample(files_chips, size=round(length(files_chips)*0.2) )}
#
# files_chips_training <- files_chips[!(files_chips %in% files_chips_samples)]
#
# file.copy(files_chips_training,
#           "E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/training/",
#           overwrite = TRUE )
#
# files_chips_training_masks <- list.files("E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/chips/masks/",
#            full.names = TRUE)
#
# files_chips_training_masks_training <- files_chips_training_masks[basename(files_chips_training_masks) %in% basename(files_chips_training)]
#
# file.copy(files_chips_training_masks_training,
#           "E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/training/masks",
#           overwrite = TRUE )
#
# ##
#
# # ### validation images
#
# {set.seed(88); files_chips_validation <- sample(files_chips_samples, size=round(length(files_chips_samples)*0.3) )}
#
# file.copy(files_chips_validation,
#           "E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/validation/images/",
#           overwrite = TRUE )
#
# files_chips_training_masks_validation <- files_chips_training_masks[basename(files_chips_training_masks) %in% basename(files_chips_validation)]
#
# file.copy(files_chips_training_masks_validation,
#           "E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/validation/masks",
#           overwrite = TRUE )
#
# ### testing
#
# files_chips_testing <- files_chips_samples[!(files_chips_samples %in% files_chips_validation)]
#
# file.copy(files_chips_testing,
#           "E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/testing/images/",
#           overwrite = TRUE )
#
# files_chips_training_masks_testing <- files_chips_training_masks[basename(files_chips_training_masks) %in% basename(files_chips_testing)]
#
# file.copy(files_chips_training_masks_testing,
#           "E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/testing/masks/",
#           overwrite = TRUE )
#
# ##

.libPaths( "C:/R/R-4.5.2/library")

library(tidyverse)
library(terra)
library(sf)
library(torch)
library(luz)
library(tmap)
library(tmaptools)
library(geodl)

trainDF <- makeChipsDF(folder="E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/training/",
                       extension=".tif",
                       mode="All",
                       shuffle=TRUE)

valDF <- makeChipsDF(folder="E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/validation/",
                     extension=".tif",
                     mode="All",
                     shuffle=TRUE)

testDF <- makeChipsDF(folder="E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/testing/",
                      extension=".tif",
                      mode="All",
                      shuffle=TRUE)

trainDS <- defineSegDataSet(chpDF=trainDF,
                            folder="E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/training/",
                            mskAdd = 1,
                            bands = c(1, 2, 3, 4),
                            doAugs = TRUE,
                            maxAugs = 3,
                            probVFlip = .5,
                            probHFlip = .5,
                            probRotation = .5)

valDS <- defineSegDataSet(chpDF=valDF,
                          folder="E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/validation/",
                          mskAdd = 1)

testDS <- defineSegDataSet(chpDF=testDF,
                           folder="E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/images/tree_spec_rast/testing/",
                           mskAdd = 1)

trainDL <- torch::dataloader(trainDS,
                             batch_size=10,
                             shuffle=TRUE,
                             drop_last = TRUE)

valDL <- torch::dataloader(valDS,
                           batch_size=10,
                           shuffle=FALSE,
                           drop_last = TRUE)

testDL <- torch::dataloader(testDS,
                            batch_size=10,
                            shuffle=FALSE,
                            drop_last = TRUE)

viewBatch(dataLoader=trainDL,
          nCols = 5,
          r = 1,
          g = 2,
          b = 3,
          cCodes=c(1, 2, 3),
          cNames=c("Background", "Conifer", "Broardleave"),
          cColors=c("gray", "red", "darkgreen")
)

trainStats <- describeBatch(trainDL,
                            zeroStart=FALSE)
valStats <- describeBatch(valDL,
                          zeroStart=FALSE)
testStats <- describeBatch(testDL,
                           zeroStart=FALSE)

### 30.01.2026 = adopt to own example (nCls = 3, device = "CPU" ...), source : https://wvview.org/gslr/c16geodl_P2.html
fitted <- defineUNet |>
  luz::setup(
    loss = defineUnifiedFocalLoss(nCls=3,
                                  lambda=0,
                                  gamma=.7,
                                  delta=0.6,
                                  smooth = 1e-8,
                                  zeroStart=FALSE,
                                  clsWghtsDist=1,
                                  clsWghtsReg=c(0.3, 0.7),
                                  useLogCosH =FALSE,
                                  device="cpu"),
    optimizer = optim_adamw,
    metrics = list(
      luz_metric_overall_accuracy(nCls=3,
                                  smooth=1,
                                  mode = "multiclass",
                                  zeroStart= FALSE),
      luz_metric_f1score(nCls=3,
                         smooth=1,
                         mode = "multiclass",
                         zeroStart= FALSE,
                         clsWghts=c(0,1)),
      luz_metric_recall(nCls=3,
                        smooth=1,
                        mode = "multiclass",
                        zeroStart= FALSE,
                        clsWghts=c(0,1)),
      luz_metric_precision(nCls=3,
                           smooth=1,
                           mode = "multiclass",
                           zeroStart= FALSE,
                           clsWghts=c(0,1))
    )
  ) |>
  set_hparams(inChn = 4,
              nCls = 3,
              actFunc = "lrelu",
              useAttn = FALSE,
              useSE = FALSE,
              useRes = TRUE,
              useASPP = TRUE,
              useDS = FALSE,
              enChn = c(16,32,64,128),
              dcChn = c(128,64,32,16),
              btnChn = 256,
              dilRates=c(1,2,4,8,16),
              dilChn=c(128,128,128,128),
              negative_slope = 0.01) |>
  set_opt_hparams(lr = 1e-3) |>
  fit(data = trainDL,
      epochs = 25,
      valid_data = valDL,
      callbacks = list(luz_callback_csv_logger("E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/unet/Logs.csv"),
                       callback_save_model_state_dict(save_dir = "E:/_Fernerkundungsprojekte/029_deepLearning_test/geodl/forest_class/indata/unet/", prefix = "epoch")),
      accelerator = accelerator(device_placement = TRUE,
                                cpu = TRUE,
                                cuda_index = torch::cuda_current_device()),
      verbose=TRUE)

################################################################################
###                               the end                                    ###
################################################################################

plotRGB(a, r=4, g=1, b=2, stretch="lin")


makeMasks(image = str_glue("{fldPth}img.tif"),
          features = str_glue("{fldPth}mskPoly.shp"),
          crop = TRUE,
          extent = str_glue("{fldPth}extent.shp"),
          field = "gridcode",
          background = 0,
          outMask = str_glue("{outPth}lcMskOut.tif"),
          mode = "Mask")

# fd ----------------------------------------------------------------------

hessen_dop_20 <-
file_temp = terra::rast(x = "E:/DeepLearning/108_LaubNadel/Daten/LaubNadel.gdb",
                        subds = 'Training_FC')

ext <- st_read(str_glue("{fldPth}extent/tree_spec_stand_vec.shp"), quiet=TRUE)


plotRGB(img, stretch="lin")

tm_shape(ext)+
  tm_borders(col="red", lwd=2)+
  tm_shape(conifers)+
  tm_polygons(fill="gray")

tm_shape(ext)+
  tm_borders(col="red", lwd=2)+
  tm_shape(lc)+
  tm_polygons(fill="lc_numb",
              fill.scale = tm_scale_categorical(labels = c("confi",
                                                           "broardlea"
              ),
              values = c("red",
                         "green")),
              fill.legend = tm_legend("Forest class Cover"))

# one class example -------------------------------------------------------

makeMasks(image = str_glue("{fldPth}images/test_clip_raster_4band.tif"),
          features = conifers,
          crop = TRUE,
          extent = ext,
          field = "lc_numb",
          background = 0,
          outImage = str_glue("{outPth}imgOut.tif"),
          outMask = str_glue("{outPth}buildingMskOut.tif"),
          mode = "Both")

img2 = rast(str_glue("{outPth}imgOut.tif"))
mskB = rast(str_glue("{outPth}buildingMskOut.tif"))

tm_shape(mskB)+
  tm_raster(col.scale = tm_scale_categorical(labels = c("Background",
                                                        "Confifers"),
                                             values = c("gray60",
                                                        "red")),
            col.legend = tm_legend("Land Cover"))


# multiple class example --------------------------------------------------

makeMasks(image = img,
          features = lc,
          crop = TRUE,
          extent = ext,
          field = "lc_numb",
          background = 0,
          outMask = str_glue("{outPth}lcMskOut.tif"),
          mode = "Mask")

mskLC = rast(str_glue("{outPth}lcMskOut.tif"))

tm_shape(mskLC)+
  tm_raster(col.scale = tm_scale_categorical(labels = c("Background",
                                                        "Conifer",
                                                        "Broardleave"),
                                             values = c("gray60",
                                                        "red",
                                                        "green")),
            col.legend = tm_legend("Land Cover"))

dir.create(str_glue("{outPth}lcChips/"))

makeChipsMultiClass(image = str_glue("{outPth}imgOut.tif"),
                    mask = str_glue("{outPth}lcMskOut.tif"),
                    n_channels = 4,
                    size = 30,
                    stride_x = 20,
                    stride_y = 20,
                    outDir = str_glue("{outPth}lcChips/"),
                    useExistingDir=FALSE)

a <- rast(str_glue("{outPth}lcChips/images/imgOut_21_21.tif"))
b <- rast(str_glue("{outPth}lcChips/masks/imgOut_21_21.tif"))
plotRGB(a)
plot(b)

lcChipsDF <- makeChipsDF(folder = str_glue("{outPth}lcChips/"),
                         extension=".tif",
                         mode="All",
                         saveCSV=FALSE)

head(lcChipsDF) |>
  gt()

viewChips(chpDF=lcChipsDF,
          folder=str_glue("{outPth}lcChips/"),
          nSamps = 10,
          mode = "both",
          justPositive = FALSE,
          cCnt = 5,
          rCnt = 2,
          r = 1,
          g = 2,
          b = 3,
          rescale = FALSE,
          rescaleVal = 1,
          cCodes = c(0,1,2),
          cNames= c("Background", "Conifers", "Broardleaves"),
          cColors= c("gray", "red", "green"),
          useSeed = TRUE,
          seed = 14)

lcDS <- defineSegDataSet(chpDF= lcChipsDF,
                         folder=str_glue("{outPth}lcChips/"),
                         normalize = FALSE,
                         rescaleFactor = 255,
                         mskRescale = 1,
                         mskAdd = 1,
                         bands = c(1, 2, 3),
                         bMns = 1,
                         bSDs = 1,
                         doAugs = TRUE,
                         maxAugs = 3,
                         probVFlip = .3,
                         probHFlip = .3,
                         probBrightness = .1,
                         probContrast = .1,
                         probGamma = 0,
                         probHue = 0,
                         probRotation = .3,
                         probSaturation = .2,
                         brightFactor = c(0.8, 1.2),
                         contrastFactor = c(0.8, 1.2),
                         gammaFactor = c(0.8, 1.2, 1),
                         hueFactor = c(-0.2, 0.2),
                         saturationFactor = c(0.8, 1.2)
)

lcDL <- torch::dataloader(lcDS,
                          batch_size=5,
                          shuffle=TRUE,
                          drop_last = TRUE)

length(lcDL)

viewBatch(lcDL,
          nCols = 3,
          r = 1,
          g = 2,
          b = 3,
          cCodes=c(1,2,3,4,5),
          cNames=c("Background", "Buildings", "Woodland", "Water", "Road"),
          cColors=c("gray60", "red", "green", "blue", "black")
)

describeBatch(lcDL,
              zeroStart=FALSE)
