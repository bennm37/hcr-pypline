import itk
import argparse

parser = argparse.ArgumentParser(description="Segment With Watershed Image Filter.")
parser.add_argument("input_image")
parser.add_argument("output_image")
parser.add_argument("threshold", type=float)
parser.add_argument("level", type=float)
args = parser.parse_args()


Dimension = 2

FloatPixelType = itk.ctype("float")
FloatImageType = itk.Image[FloatPixelType, Dimension]

reader = itk.ImageFileReader[FloatImageType].New()
reader.SetFileName(args.input_image)

gradientMagnitude = itk.GradientMagnitudeImageFilter.New(Input=reader.GetOutput())

watershed = itk.WatershedImageFilter.New(Input=gradientMagnitude.GetOutput())

threshold = args.threshold
level = args.level
watershed.SetThreshold(threshold)
watershed.SetLevel(level)

LabeledImageType = type(watershed.GetOutput())

PixelType = itk.ctype("unsigned char")
RGBPixelType = itk.RGBPixel[PixelType]
RGBImageType = itk.Image[RGBPixelType, Dimension]

ScalarToRGBColormapFilterType = itk.ScalarToRGBColormapImageFilter[
    LabeledImageType, RGBImageType
]
colormapImageFilter = ScalarToRGBColormapFilterType.New()
colormapImageFilter.SetColormap(
    itk.ScalarToRGBColormapImageFilterEnums.RGBColormapFilter_Jet
)
colormapImageFilter.SetInput(watershed.GetOutput())
colormapImageFilter.Update()

WriterType = itk.ImageFileWriter[RGBImageType]
writer = WriterType.New()
writer.SetFileName(args.output_image)
writer.SetInput(colormapImageFilter.GetOutput())
writer.Update()