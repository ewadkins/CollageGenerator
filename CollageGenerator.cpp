#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>


std::vector<std::string> cleanMap(std::vector<std::string> lines) {
	std::vector<std::vector<bool> > map;
	std::vector<std::string> cleaned;
	const int borderSize = 1;
	int maxLength = 0;
	for (int i = 0; i < lines.size(); i++) {
		if (lines[i].length() > maxLength) {
			maxLength = lines[i].length();
		}
	}
	for (int i = 0; i < lines.size(); i++) {
		while (lines[i].length() < maxLength) {
			lines[i] += " ";
		}
		for (int j = 0; j < borderSize; j++) {
			lines[i] = " " + lines[i] + " ";
		}
	}
	std::string extraLine = "";
	for (int i = 0; i < maxLength + borderSize * 2; i++) {
		extraLine += " ";
	}
	for (int i = 0; i < borderSize; i++) {
		cleaned.push_back(extraLine);
	}
	for (int i = 0; i < lines.size(); i++) {
		cleaned.push_back(lines[i]);
	}
	for (int i = 0; i < borderSize; i++) {
		cleaned.push_back(extraLine);
	}
	return cleaned;
}

std::vector<std::string> loadMapFromFile(const char* filepath) {
	std::ifstream stream(filepath);
	std::vector<std::string> lines;
	std::string line;
	while(std::getline(stream, line)) {
		if (line[line.length() - 1] == '\r') {
			line = line.substr(0, line.length() - 1);
		}
		lines.push_back(line);
	}
	return cleanMap(lines);
}

std::vector<std::string> loadMapFromString(const char* string) {
	std::istringstream stream(string);
	std::vector<std::string> lines;
	std::string line;
	while(std::getline(stream, line)) {
		if (line[line.length() - 1] == '\r') {
			line = line.substr(0, line.length() - 1);
		}
		lines.push_back(line);
	}
	return cleanMap(lines);
}

std::vector<std::vector<bool> > parseMap(std::vector<std::string> lines) {
	std::vector<std::vector<bool> > map;
	for (int i = 0; i < lines.size(); i++) {
		std::vector<bool> vec;
		for (int j = 0; j < lines[i].length(); j++) {
			if (lines[i][j] != ' ') {
				vec.push_back(true);
			}
			else {
				vec.push_back(false);
			}
		}
		map.push_back(vec);
	}
	return map;
}

cv::Mat makeCanvas(std::vector<cv::Mat> images, const int size, const int imagesPerRow) {
	int width = size * std::fmin(images.size(), imagesPerRow);
	int height = size * (images.size() / imagesPerRow + (images.size() % imagesPerRow > 0 ? 1 : 0));
	cv::Mat canvas = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

	for (int i = 0; i < images.size(); i++) {
		cv::Mat img = images[i];
		cv::resize(img, img, cv::Size(size, size));
		cv::Mat canvasROI(canvas, cv::Rect(size * (i % imagesPerRow),
				size * (i / imagesPerRow), size, size));
		img.copyTo(canvasROI);
	}

	return canvas;
}

cv::Mat makeCollage(std::vector<std::vector<bool> > map, std::vector<cv::Mat> primary, std::vector<cv::Mat> secondary, int size) {
	std::vector<cv::Mat> images;
	for (int i = 0; i < map.size(); i++) {
		for (int j = 0; j < map[0].size(); j++) {
			images.push_back(map[i][j] ?
					primary[rand() % primary.size()] : secondary[rand() % secondary.size()]);
		}
	}
	cv::Mat collage = makeCanvas(images, size, map[0].size());
	return collage;
}

double colorDistance(cv::Vec3b color1, cv::Vec3b color2) {
	return std::sqrt(std::pow((double) (color2[0] - color1[0]), 2)
					+ std::pow((double) (color2[1] - color1[1]), 2)
					+ std::pow((double) (color2[2] - color1[2]), 2));
}

cv::Mat kMeansColorMap(cv::Mat src, const int clusterCount, const int attempts = 1) {
	cv::Mat samples(src.rows * src.cols, 3, CV_32F);
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x * src.rows, z) = src.at<cv::Vec3b>(y, x)[z];

	cv::Mat labels;
	cv::Mat centers;
	cv::kmeans(samples, clusterCount, labels,
			cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001),
			attempts, cv::KMEANS_PP_CENTERS, centers);

	cv::Mat newImg(src.size(), src.type());
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++) {
			int cluster_idx = labels.at<int>(y + x * src.rows, 0);
			newImg.at<cv::Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			newImg.at<cv::Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			newImg.at<cv::Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	return newImg;
}

std::pair<std::vector<cv::Vec3b>, std::vector<int> > kMeansCluster(cv::Mat src, const int clusterCount, const int attempts = 1) {
	cv::Mat samples(src.rows * src.cols, 3, CV_32F);
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x * src.rows, z) = src.at<cv::Vec3b>(y, x)[z];

	cv::Mat labels;
	cv::Mat centers;
	cv::kmeans(samples, clusterCount, labels,
			cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001),
			attempts, cv::KMEANS_PP_CENTERS, centers);

	std::vector<cv::Vec3b> colors;
	for (int i = 0; i < clusterCount; i++) {
		cv::Vec3b color(centers.at<float>(i, 0),
						centers.at<float>(i, 1),
						centers.at<float>(i, 2));
		colors.push_back(color);
	}

	std::vector<int> ids;
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			ids.push_back(labels.at<int>(y + x * src.rows, 0));
		}
	}
	return std::pair<std::vector<cv::Vec3b>, std::vector<int> >(colors, ids);
}

int main(int argc,char *argv[]) {
	if (argc < 4) {
		std::cout << "Usage: ./CollageGenerator <image directory> <input file> <tile size> [output file (displays if omitted)]" << std::endl;
		exit(1);
	}

	std::string imageDirectoryPath = argv[1];
	std::string inputFilePath = argv[2];
	const int tileSize = atoi(argv[3]);
	std::string outputFilePath = "";
	if (argc >= 5) {
		outputFilePath = argv[4];
	}

	std::cout << "Image directory: " << imageDirectoryPath << std::endl;
	std::cout << "Input file: " << inputFilePath << std::endl;
	std::cout << "Tile size: " << tileSize << std::endl;
	std::cout << "Output file: " << (outputFilePath != "" ? outputFilePath : "None, displaying instead") << std::endl;

	if (imageDirectoryPath[imageDirectoryPath.length() - 1] != '/') {
		imageDirectoryPath += "/";
	}

	std::vector<std::string> lines = loadMapFromFile(inputFilePath.c_str());

	for (int i = 0; i < lines.size(); i++) {
		std::cout << lines[i] << std::endl;
	}

	std::vector<std::vector<bool> > map = parseMap(lines);

	std::string imagesPath = imageDirectoryPath + "*.jpg";

	std::vector<cv::String> filenames;
	cv::glob(imagesPath, filenames);
	std::vector<cv::Mat> images;
	std::vector<cv::Mat> averagedImages;


	for (int i = 0; i < filenames.size(); i++) {
		std::cout << "PROGRESS " << (i + 1) << "/" << filenames.size() << " (" << (i + 1) * 100 / filenames.size() << "%)" << std::endl;
		cv::Mat img = cv::imread(filenames[i]);
		cv::Scalar average = cv::mean(img);
		cv::Mat averaged(1, 1, img.type());
		averaged.setTo(cv::Vec3b(average[0], average[1], average[2]));

		images.push_back(img);
		averagedImages.push_back(averaged);
	}


	const bool DEBUG = false;
	const int debugImagesPerRow = std::ceil(std::sqrt(images.size()));
	const int debugSize = 350 / debugImagesPerRow;

	const int clusterCount = std::fmax(2, std::log2(images.size()));
	const int attempts = 3;


	cv::Mat averagedCanvas = makeCanvas(averagedImages, 1, images.size());
	cv::Mat clustered = kMeansColorMap(averagedCanvas, clusterCount, attempts);
	std::pair<std::vector<cv::Vec3b>, std::vector<int> > clusterData = kMeansCluster(clustered, clusterCount, attempts);
	std::vector<cv::Vec3b> colors = clusterData.first;
	std::vector<int> clusterIds = clusterData.second;

	std::vector<int> ids;
	for (int i = 0; i < images.size(); i++) {
		ids.push_back(clusterIds[i]);
		//std::cout << clusterIds[i] << std::endl;
	}

	std::vector<int> averageColor(3);
	for (int i = 0; i < ids.size(); i++) {
    	averageColor[0] += colors[ids[i]][0];
    	averageColor[1] += colors[ids[i]][1];
    	averageColor[2] += colors[ids[i]][2];
	}
	averageColor[0] /= ids.size();
	averageColor[1] /= ids.size();
	averageColor[2] /= ids.size();

	std::vector<int> idCount(clusterCount);
	for (int i = 0; i < ids.size(); i++) {
		idCount[ids[i]] += 1;
	}
	std::vector<double> idDistance(clusterCount);
	for (int i = 0; i < ids.size(); i++) {
		double dist = colorDistance(colors[ids[i]], cv::Vec3b(averageColor[0], averageColor[1], averageColor[2]));
		idDistance[ids[i]] = dist;
	}
	std::vector<std::pair<int, int> > countPairs;
	for (int i = 0; i < idCount.size(); i++) {
		countPairs.push_back(std::pair<int, int>(idCount[i], i));
		//std::cout << i << " count: " << idCount[i] << std::endl;
	}
	std::vector<std::pair<double, int> > distancePairs;
	for (int i = 0; i < idDistance.size(); i++) {
		distancePairs.push_back(std::pair<double, int>(idDistance[i], i));
		//std::cout << i << " dist: " << idDistance[i] << std::endl;
	}
	std::sort(countPairs.begin(), countPairs.end());
	std::sort(distancePairs.begin(), distancePairs.end());
	std::vector<int> countOrderedIds;
	std::vector<int> distanceOrderedIds;
	for (int i = countPairs.size() - 1; i >= 0; i--) {
		countOrderedIds.push_back(countPairs[i].second);
	}
	for (int i = 0; i < distancePairs.size(); i++) {
		distanceOrderedIds.push_back(distancePairs[i].second);
	}

	std::vector<cv::Mat> primary;
	std::vector<cv::Mat> secondary;
	for (int i = 0; i < images.size(); i++) {
		bool include = false;
		//for (int j = distanceOrderedIds.size() * 3/4; j < distanceOrderedIds.size(); j++) {
		std::vector<int> includedIds;
		/*int count = 0;
		for (int j = distanceOrderedIds.size() - 1; j >= 0; j--) {
			if (count < std::log2(images.size()) && count <= images.size() / 2) {
				includedIds.push_back(distanceOrderedIds[j]);
				count += idCount[distanceOrderedIds[j]];
				std::cout << "Added id " << distanceOrderedIds[j] << " count: " << count << std::endl;
			}
		}*/
		for (int j = distanceOrderedIds.size() - 1; j >= 0; j--) {
			if (idCount[distanceOrderedIds[j]] >= std::log2(images.size())) {
				includedIds.push_back(distanceOrderedIds[j]);
				break;
			}
		}
		if (includedIds.size() == 0) {
			includedIds.push_back(distanceOrderedIds[distanceOrderedIds.size() - 1]);
		}
		for (int j = 0; j < includedIds.size(); j++) {
			if (ids[i] == includedIds[j]) {
				include = true;
			}
		}
		if (include) {
			//cv::Mat weightedImg;
			cv::Mat img = images[i];
			//cv::Mat pink(img.rows, img.cols, CV_8UC3, cv::Scalar(128, 128, 255));
			//cv::addWeighted(img, 0.6, pink, 0.4, 0.0, weightedImg);
			primary.push_back(img);
		}
		else {
			secondary.push_back(images[i]);
		}
	}
	/*for (int i = 0; i < images.size(); i++) {
		bool include = false;
		//for (int j = distanceOrderedIds.size() * 3/4; j < distanceOrderedIds.size(); j++) {
		for (int j = distanceOrderedIds.size() - 1; j < distanceOrderedIds.size(); j++) {
			if (ids[i] == distanceOrderedIds[j]) {
				include = true;
			}
		}
		if (include) {
			primary.push_back(images[i]);
		}
		else {
			secondary.push_back(images[i]);
		}
	}*/

	cv::Mat collage = makeCollage(map, primary, secondary, tileSize);

	if (DEBUG) {
		cv::Mat imagesCanvas = makeCanvas(images, debugSize, debugImagesPerRow);
		cv::Mat debugAveragedCanvas = makeCanvas(averagedImages, debugSize, debugImagesPerRow);
		cv::Mat debugClusteredCanvas = kMeansColorMap(debugAveragedCanvas, clusterCount, attempts);
		cv::Mat primaryCanvas = makeCanvas(primary, debugSize, debugImagesPerRow);
		cv::Mat secondaryCanvas = makeCanvas(secondary, debugSize, debugImagesPerRow);
		std::vector<cv::Mat> clusterImages;
		clusterImages.push_back(cv::Mat(1, 1, CV_8UC3, cv::Vec3b(averageColor[0], averageColor[1], averageColor[2])));
		for (int i = 0; i < colors.size(); i++) {
			cv::Mat clusterImg(1, 1, CV_8UC3, colors[i]);
			clusterImages.push_back(clusterImg);
		}
		cv::Mat clustersCanvas = makeCanvas(clusterImages, 30, clusterImages.size() + 1);
		cv::imshow("images", imagesCanvas);
		cv::imshow("averaged", debugAveragedCanvas);
		cv::imshow("clustered", debugClusteredCanvas);
		cv::imshow("primary", primaryCanvas);
		cv::imshow("secondary", secondaryCanvas);
		cv::imshow("clusters", clustersCanvas);
	}

	if (outputFilePath != "") {
		cv::imwrite(outputFilePath, collage);
	}
	else {
		std::cout << "Press ESC to exit" << std::endl;
		cv::imshow("collage", collage);
		cv::waitKey(0);
	}
}
