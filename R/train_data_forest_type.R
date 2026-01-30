
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

  ext_loop <- ext %>%         # choose extent
    filter(id_img == image_id)

  plot(st_geometry(ext_loop)) # plot extent

  makeChipsMultiClass(image = image_folder,
                      mask =  paste0(image_folder, "masks"),
                      n_channels = 4,
                      size = 512,
                      stride_x = 512,
                      stride_y = 512,
                      outDir = paste0(image_folder, "chips"),
                      useExistingDir=FALSE)

}





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
