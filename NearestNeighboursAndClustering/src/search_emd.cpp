#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <tuple>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/signal.h>
#include <queue>
#include "../headers/dataset.h"
#include "../headers/labelDataset.h"
#include "../headers/image.h"
#include "../headers/bruteforce_search.h"
#include "../headers/utilities.h"

const std::string write_pipe = "write_fifo", read_pipe = "read_fifo";

void finish_execution(int signum) {
    signal(SIGINT, finish_execution);
    signal(SIGQUIT, finish_execution);
    unlink(write_pipe.c_str());
    unlink(read_pipe.c_str());
    exit(1);
}

double eucledian_distance(int x1,int y1,int x2,int y2) {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

void usage() {
    std::cout << "Usage:./search_emd -d <input  file  original  space> -q <query  file  original  space> -l1 <labels of input dataset> -l2 <labels of query dataset> -o <output file>";
}

int main(int argc, char const *argv[]) {
    // Register signal handler
    signal(SIGINT,finish_execution);
    signal(SIGQUIT,finish_execution);

  	std::string inputFileOriginalSpace, queryFileOriginalSpace, inputDatasetLabels, queryDatasetLabels, outputFile;

    // Check usage
    if (argc == 11) {
        for(int i = 1; i < argc; i+=2) {
            std::string arg(argv[i]);

            if(!arg.compare("-d")) {
                inputFileOriginalSpace = argv[i+1];
            }
            else if(!arg.compare("-q")) {
                queryFileOriginalSpace = argv[i+1];
            }
            else if(!arg.compare("-l1")) {
				inputDatasetLabels = argv[i+1];
			}
            else if(!arg.compare("-l2")) {
				queryDatasetLabels = argv[i+1];
			}
			else if(!arg.compare("-o")) {
				outputFile = argv[i+1];
			}
			else {
				usage();
				return 0;
			}
        }
    }
	else {
		usage();
		return 0;
	}

    std::ofstream outputStream(outputFile);

    // Create named pipes to communicate with the emd calculator program
    if(mkfifo(write_pipe.c_str(), 0666) < 0 || mkfifo(read_pipe.c_str(), 0666) < 0) {
        perror("mkfifo");
        return 1;
    }

    // Load training dataset
    Dataset<Pixel8Bit> *trainDataset = new Dataset<Pixel8Bit>(inputFileOriginalSpace);
    std::vector<Image<Pixel8Bit>*> trainImages = trainDataset->getImages();

    // Load training labels
    LabelDataset *trainLabelDataset = new LabelDataset(inputDatasetLabels);

    // Load query dataset
    Dataset<Pixel8Bit> *queryDataset = new Dataset<Pixel8Bit>(queryFileOriginalSpace);
    std::vector<Image<Pixel8Bit>*> queryImages = queryDataset->getImages();

    // Load query labels
    LabelDataset *queryLabelDataset = new LabelDataset(queryDatasetLabels);

    // Initialize bruteforce searcher for manhattan
    Bruteforce_Search<Pixel8Bit> *bruteforce = new Bruteforce_Search<Pixel8Bit>(trainImages);

    int cluster_dimension = 7;
    int clusteredSize = trainDataset->getImageDimension() / (cluster_dimension * cluster_dimension);
    double d[clusteredSize][clusteredSize];
    double wi[clusteredSize];
    double wj[clusteredSize];
    double queryTotalValue,imgTotalValue;
    double emd;

    int pid = fork();
    if(pid == 0) {
        execl("./or_tools/bin/emd","emd", write_pipe.c_str(), read_pipe.c_str(), std::to_string(clusteredSize).c_str(), NULL);
        perror("execl");
        exit(1);
    }
    else if(pid == -1) {
        std::cerr << "- Error: Fork failed" << std::endl;
        return 1;
    }

    int writeFd = open(write_pipe.c_str(), O_WRONLY);
    int readFd = open(read_pipe.c_str(), O_RDONLY);

    // Calculate clusters for all train images
    std::vector<std::vector<Image<Pixel8Bit>*>> imageClusters;
    
    double emd_average_correct_search_results = 0.0;
    double manhattan_correct_search_results = 0.0;
    for (std::vector<Image<Pixel8Bit>*>::iterator qit = queryImages.begin();qit != queryImages.begin()+1; qit++) {
        //std::cout << "Query: " << (*qit)->getId() << std::endl;

        // Generate query image clusters
        std::vector<Image<Pixel8Bit>*> queryClusters = (*qit)->findClusters(cluster_dimension);
        queryTotalValue = (*qit)->totalValue();

        std::priority_queue<std::pair<double,int>,std::vector<std::pair<double,int>>,std::greater<std::pair<double,int>>> prQueue;
        
        for (std::vector<Image<Pixel8Bit>*>::iterator tit = trainImages.begin(); tit != trainImages.end(); tit++) {
            unsigned int img_index = tit - trainImages.begin();
            if(img_index >= imageClusters.size())
                imageClusters.push_back((*tit)->findClusters(cluster_dimension));

            // Calculate normalized cluster weights
            imgTotalValue = (*tit)->totalValue();
            for (unsigned int i = 0; i < imageClusters[img_index].size(); i++) {
                wi[i] = imageClusters[img_index][i]->totalValue() / imgTotalValue;

                std::tuple<int,int> imageCentroid = imageClusters[img_index][i]->findCentroid();
                int x1 = (imageClusters[img_index][i]->getId() % ((*tit)->getWidth() / imageClusters[img_index][i]->getWidth())) * imageClusters[img_index][i]->getWidth() + std::get<0>(imageCentroid);
                int y1 = (imageClusters[img_index][i]->getId() / ((*tit)->getWidth() / imageClusters[img_index][i]->getWidth())) * imageClusters[img_index][i]->getHeight() + std::get<1>(imageCentroid);
                
                for (unsigned int j = 0; j < queryClusters.size(); j++) {
                    wj[j] = queryClusters[j]->totalValue() / queryTotalValue;

                    std::tuple<int,int> queryCentroid = queryClusters[j]->findCentroid();
                    int x2 = (queryClusters[j]->getId() % ((*qit)->getWidth() / queryClusters[j]->getWidth())) * queryClusters[j]->getWidth() + std::get<0>(queryCentroid);
                    int y2 = (queryClusters[j]->getId() / ((*qit)->getWidth() / queryClusters[j]->getWidth())) * queryClusters[j]->getHeight() + std::get<1>(queryCentroid);

                    d[i][j] = eucledian_distance(x1, y1, x2, y2);
                }
            }

            // Send the arrays
            if(write(writeFd, d, sizeof(d)) <= 0 || write(writeFd, wi, sizeof(wi)) <= 0 || write(writeFd, wj, sizeof(wj)) <= 0) {
                perror("write");
                break;
            }
            // Receive the result
            if(read(readFd, &emd, sizeof(emd)) <= 0) {
                perror("read");
                break;
            }

            // Insert it to the priority queue
            prQueue.push(std::pair<double,int>(emd,img_index));

            //std::cout << "EMD to Image " << (*tit)->getId() << ": " << emd << std::endl;
        }
        //std::cout << std::endl;
        unsigned int queryIndex = qit - queryImages.begin();

        // Find mannhattan 10 Nearest Neighbours
        std::vector<std::pair<double, int>> mannhattanNNeighbours = bruteforce->exactNN(*qit,10);
        int correctNNs = 0;
        
        // Compare their labels with the correct one
        for (unsigned int i = 0;i < mannhattanNNeighbours.size();i++) {
            if (queryLabelDataset->getLabel(queryIndex) == trainLabelDataset->getLabel(mannhattanNNeighbours[i].second)) {
                correctNNs++;
            }
        }
        manhattan_correct_search_results += correctNNs/mannhattanNNeighbours.size();

        // Find EMD 10 Nearest Neighbours
        correctNNs = 0;
        for (unsigned int i = 0;i < 10;i++) {
            if (queryLabelDataset->getLabel(queryIndex) == trainLabelDataset->getLabel(prQueue.top().second)) {
                correctNNs++;
            }
            prQueue.pop();
        }
        emd_average_correct_search_results += correctNNs/10;

        for (unsigned int i = 0; i < queryClusters.size(); i++) {
            delete queryClusters[i];
        }
    }

    outputStream << "Average Correct Search Results EMD: " << emd_average_correct_search_results / 1 << std::endl;
    outputStream << "Average Correct Search Results MANHATTAN: " << manhattan_correct_search_results / 1 << std::endl;
    outputStream.close();

    for (unsigned int i = 0; i < imageClusters.size(); i++)
        for (unsigned int j = 0; j < imageClusters[i].size(); j++)
            delete imageClusters[i][j];

    close(readFd);
    close(writeFd);

    int status = 0;
    while (wait(&status) > 0);

    unlink(write_pipe.c_str());
    unlink(read_pipe.c_str());
    
    delete bruteforce;
    delete queryLabelDataset;
    delete queryDataset;
    delete trainLabelDataset;
    delete trainDataset;
    
    return 0;
}
