
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include "../headers/dataset.h"
#include "../headers/bruteforce_search.h"
#include "../headers/lsh.h"

void usage() {
    std::cout << "Usage:./search –d <input file original space> –i <input file new space> -q <query file original space> -s <query file new space>  –k <int>  -L <int> -ο  <output file>\n";
}

int main(int argc, char const *argv[])
{
	std::string inputFileOriginalSpace, inputFileNewSpace, queryFileOriginalSpace, queryFileNewSpace, outputFile;
    // Set default parameter values
	int k = 4, L = 5, N = 1;

    // Check usage
    if (argc == 15) {
        for(int i = 1; i < argc; i+=2) {
            std::string arg(argv[i]);

            if(!arg.compare("-d")) {
                inputFileOriginalSpace = argv[i+1];
            }
            else if(!arg.compare("-i")) {
                inputFileNewSpace = argv[i+1];
            }
            else if(!arg.compare("-q")) {
				queryFileOriginalSpace = argv[i+1];
			}
            else if(!arg.compare("-s")) {
				queryFileNewSpace = argv[i+1];
			}
			else if(!arg.compare("-k")) {
				k = atoi(argv[i+1]);
			}
			else if(!arg.compare("-L")) {
				L = atoi(argv[i+1]);
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

    // Read Dataset
    Dataset<Image,Pixel8Bit> *datasetOriginalSpace = new Dataset<Image,Pixel8Bit>(inputFileOriginalSpace);
    Dataset<ImageReduced,Pixel16Bit> *datasetNewSpace = new Dataset<ImageReduced,Pixel16Bit>(inputFileNewSpace);

    if (datasetOriginalSpace->isValid() && datasetNewSpace->isValid()) {
        // Get images from datasets
        std::vector<Image*> imagesOriginalSpace = datasetOriginalSpace->getImages();
        std::vector<ImageReduced*> imagesNewSpace = datasetNewSpace->getImages();

        // Initialize LSH interface
        int w = datasetOriginalSpace->avg_NN_distance() * 6;
        LSH *lsh = new LSH(k, w, L, datasetOriginalSpace);

        Bruteforce_Search<Image> *bruteforceOriginalSpace = new Bruteforce_Search<Image>(imagesOriginalSpace);
        Bruteforce_Search<ImageReduced> *bruteforceNewSpace = new Bruteforce_Search<ImageReduced>(imagesNewSpace);

        // Open output file
        std::ofstream outputStream(outputFile);
        
        bool repeat;
        do {
            // Read querysets
            Dataset<Image,Pixel8Bit> *querysetOriginalSpace = new Dataset<Image,Pixel8Bit>(queryFileOriginalSpace);
            Dataset<ImageReduced,Pixel16Bit> *querysetNewSpace = new Dataset<ImageReduced,Pixel16Bit>(queryFileNewSpace);

            if (!querysetOriginalSpace->isValid()) {
                delete querysetOriginalSpace;
                break;
            }

            // Get query images
            std::vector<Image*> queryImagesOriginalSpace = querysetOriginalSpace->getImages();
            std::vector<ImageReduced*> queryImagesNewSpace = querysetNewSpace->getImages();
            double total_bf_time=0.0,total_lsh_time=0.0,total_reduced_time=0.0;
            for (unsigned int i = 0; i < queryImagesOriginalSpace.size(); i++) {
                outputStream << "Query: " << queryImagesOriginalSpace[i]->getId() << std::endl;

                clock_t begin_reduced_time = clock();
                std::vector<std::pair<double, int>> reducedNearestNeighbours = bruteforceNewSpace->exactNN(queryImagesNewSpace[i],N);
                double reduced_time = double(clock() - begin_reduced_time) / CLOCKS_PER_SEC;
                total_reduced_time += reduced_time;

                clock_t begin_lsh_time = clock();
                std::vector<std::pair<double, int>> lshNearestNeighbours = lsh->approximate_kNN(queryImagesOriginalSpace[i],N);
                double lsh_time = double(clock() - begin_lsh_time) / CLOCKS_PER_SEC;
                total_lsh_time += lsh_time;

                clock_t begin_bf_time = clock();
                std::vector<std::pair<double, int>> exactNearestNeighbours = bruteforceOriginalSpace->exactNN(queryImagesOriginalSpace[i],N);
                double bf_time = double(clock() - begin_bf_time) / CLOCKS_PER_SEC;
                total_bf_time += bf_time;
                
                outputStream << "Nearest neighbor Reduced: " << reducedNearestNeighbours[0].second << std::endl;
                outputStream << "Nearest neighbor LSH: " << lshNearestNeighbours[0].second << std::endl;
                outputStream << "Nearest neighbor True: " << exactNearestNeighbours[0].second << std::endl;
                outputStream << "distanceReduced: " << reducedNearestNeighbours[0].first << std::endl;
                outputStream << "distanceLSH: " << lshNearestNeighbours[0].first << std::endl;
                outputStream << "distanceTrue: " << exactNearestNeighbours[0].first << std::endl;
            }

            outputStream << "tReduced: " << total_reduced_time << std::endl;
            outputStream << "tLSH: " << total_lsh_time << std::endl;
            outputStream << "tTrue: " << total_bf_time << std::endl;
            
            outputStream << std::endl;
            
            delete querysetOriginalSpace;
            delete querysetNewSpace;
            
            std::string promptAnswer;
            std::cout << "Do you want to enter another query (Y/N)?  ";

            do {
                std::cin >> promptAnswer;
            } while(promptAnswer != "Y" && promptAnswer != "N" && std::cout << "Invalid answer" << std::endl);

            repeat = (promptAnswer == "Y");

            if(repeat) {
                std::cout << "Query File Original Space: ";
                std::cin >> queryFileOriginalSpace;
                std::cout << "Query File New Space: ";
                std::cin >> queryFileNewSpace;
            }

        } while(repeat);

        outputStream.close();

        delete lsh;
        delete bruteforceOriginalSpace;
        delete bruteforceNewSpace;
    } 

    delete datasetOriginalSpace;
    delete datasetNewSpace;

    return 0;
}
