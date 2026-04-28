// Package imagenet provides an interface to read the imagenet interface.
package imagenet

import (
	"bufio"
	"flag"
	"fmt"
	"image"
	"os"
	"path"
	"runtime"
	"strings"

	"github.com/disintegration/imaging"
)

var imagenetDataFolder = flag.String("imagenet-data-folder", "",
	"Specifies where the imagenet data is located at.")
var imagenetInputSize = flag.Int("img-size", 96,
	"Input image resolution (square) used by ImageNet loader and VGG16.")
var imagenetTrainClassCount = flag.Int("img-tr-classes", 200,
	"Number of training classes to iterate through.")
var imagenetTrainImagesPerClass = flag.Int("img-tr-per-class", 500,
	"Number of training images to iterate per class.")
var imagenetValImageCount = flag.Int("img-val-count", 10000,
	"Number of validation images to iterate through.")

// The DataSet can provide imagenet data.
type DataSet struct {
	mapClassToLabel map[string]int
	mapTestToClass  map[string]string
	classes         []string
	curClass        int
	curImage        int
	isTrain         bool
}

// NewDataSet creates a new imagenet dataset.
func NewDataSet(isTrain bool) *DataSet {
	fileDir := getDirPath()
	d := &DataSet{
		curClass:        0,
		curImage:        0,
		isTrain:         isTrain,
		mapClassToLabel: make(map[string]int),
		mapTestToClass:  make(map[string]string),
	}

	// read wnids.txt
	classPath := path.Join(fileDir + "/wnids.txt")
	file, err := os.Open(classPath)
	dieOnErr(err)

	scanner := bufio.NewScanner(file)
	i := 0
	for scanner.Scan() {
		line := scanner.Text()
		className := strings.Trim(line, "\n")

		// map of class to label is stored in mapClassToLabel
		d.mapClassToLabel[className] = i
		d.classes = append(d.classes, className)
	}

	err = file.Close()
	if err != nil {
		panic(err)
	}

	if !isTrain {
		// read val_annotation.txt
		testPath := path.Join(fileDir + "/val/val_annotations.txt")
		file, err = os.Open(testPath)
		dieOnErr(err)
		defer file.Close()

		scanner = bufio.NewScanner(file)
		for scanner.Scan() {
			line := scanner.Text()
			line = strings.Trim(line, "\n")
			tokens := strings.Split(line, "\t")[:2]

			d.mapTestToClass[tokens[0]] = tokens[1]
		}
	}

	return d
}

// HasNext returns true if there are more images in the dataset.
func (d *DataSet) HasNext() bool {
	trainClassCount := clampPositive(*imagenetTrainClassCount)
	trainImagesPerClass := clampPositive(*imagenetTrainImagesPerClass)
	valImageCount := clampPositive(*imagenetValImageCount)

	if d.isTrain {
		return d.curClass < trainClassCount && d.curImage < trainImagesPerClass
	}

	return d.curImage < valImageCount
}

// Next returns the next image and label. It panics if there is no more images
// in the dataset.
func (d *DataSet) Next() (imageData []byte, labelData byte) {
	var err error
	imageSize := InputImageSize()
	imageArea := imageSize * imageSize
	imageArray := make([]byte, imageArea*3)
	var label byte
	var imagePath string
	var className string

	fileDir := getDirPath()

	if d.isTrain {
		className = d.classes[d.curClass]
		imagePath = path.Join(fileDir + fmt.Sprintf("/train/%s/images/%s_%d.JPEG", className, className, d.curImage))
	} else {
		imagePath = path.Join(fileDir + fmt.Sprintf("/val/images/val_%d.JPEG", d.curImage))
	}

	f, err := os.Open(imagePath)
	dieOnErr(err)
	defer f.Close()

	img, _, err := image.Decode(f)
	resized := imaging.Resize(img, imageSize, imageSize, imaging.Lanczos)
	dieOnErr(err)
	for i := 0; i < imageSize; i++ {
		for j := 0; j < imageSize; j++ {
			r, g, b, _ := resized.At(i, j).RGBA()
			idx := i*imageSize + j
			imageArray[idx] = uint8(r >> 8)
			imageArray[idx+imageArea] = uint8(g >> 8)
			imageArray[idx+imageArea*2] = uint8(b >> 8)
		}
	}

	if d.isTrain {
		label = uint8(d.mapClassToLabel[className])
		trainImagesPerClass := clampPositive(*imagenetTrainImagesPerClass)

		d.curImage++
		if d.curImage >= trainImagesPerClass {
			d.curClass++
			d.curImage = 0
		}
	} else {
		class := d.mapTestToClass[fmt.Sprintf("val_%d.JPEG", d.curImage)]
		label = uint8(d.mapClassToLabel[class])
		d.curImage++
	}

	return imageArray, label
}

// Reset lets the dataset to read from the beginning.
func (d *DataSet) Reset() {
	d.curClass = 0
	d.curImage = 0
}

func getDirPath() string {
	if *imagenetDataFolder != "" {
		return *imagenetDataFolder
	}

	_, filename, _, ok := runtime.Caller(1)
	if !ok {
		panic("impossible")
	}

	fileDir := path.Dir(filename) + "/data/tiny-imagenet-200/"
	return fileDir
}

func dieOnErr(err error) {
	if err != nil {
		panic(err)
	}
}

func clampPositive(v int) int {
	if v <= 0 {
		return 1
	}

	return v
}

// InputImageSize returns the configured square image size.
func InputImageSize() int {
	return clampPositive(*imagenetInputSize)
}
