
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <unistd.h>
#include "../headers/dataset.h"
#include "../headers/bruteforce_search.h"
#include "../headers/lsh.h"

void usage() {
    std::cout << "Usage:./search –d <input file original space> -i <input file new space> –q <query file original space> -s <query file new space> –k <int> -L <int> -ο <output file> " << std::endl
                    << "./search –d <input file original space> –q <query file original space> -l1 <labels of input dataset> -l2 <labels of query dataset> -ο <output file> -EMD" << std::endl;
}

int main(int argc, char const *argv[])
{
	std::string inputFileOriginalSpace, inputFileNewSpace, queryFileOriginalSpace, queryFileNewSpace, outputFile, inputDatasetLabels, queryDatasetLabels;
    // Set default parameter values
	int k = 4, L = 5, N = 1;
    bool use_emd = false;

    // Check usage
    if (argc == 15 || argc == 20) {
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
    else if(argc == 12) {
        for(int i = 1; i < argc; i+=2) {
            std::string arg(argv[i]);

            if(!arg.compare("-d")) {
                inputFileOriginalSpace = argv[i+1];
            }
            else if(!arg.compare("-q")) {
				queryFileOriginalSpace = argv[i+1];
			}
			else if(!arg.compare("-o")) {
				outputFile = argv[i+1];
			}
            else if(!arg.compare("-l1")) {
				inputDatasetLabels = argv[i+1];
			}
            else if(!arg.compare("-l2")) {
				queryDatasetLabels = argv[i+1];
			}
            else if(!arg.compare("-EMD")) {
				i--;
                use_emd = true;
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

    if(use_emd) {
        execl("./search_emd","search_emd", "-d", inputFileOriginalSpace.c_str(), "-q", queryFileOriginalSpace.c_str(), "-l1", inputDatasetLabels.c_str(), "-l2", queryDatasetLabels.c_str(), "-o", outputFile.c_str(), NULL);
		perror("execl");
        exit(1);
    }

    // Read Dataset
    Dataset<Pixel8Bit> *datasetOriginalSpace = new Dataset<Pixel8Bit>(inputFileOriginalSpace);
    Dataset<Pixel16Bit> *datasetNewSpace = new Dataset<Pixel16Bit>(inputFileNewSpace);

    if (datasetOriginalSpace->isValid() && datasetNewSpace->isValid()) {
        // Get images from datasets
        std::vector<Image<Pixel8Bit>*> imagesOriginalSpace = datasetOriginalSpace->getImages();
        std::vector<Image<Pixel16Bit>*> imagesNewSpace = datasetNewSpace->getImages();

        // Initialize LSH interface
        int w = datasetOriginalSpace->avg_NN_distance() * 6;
        LSH *lsh = new LSH(k, w, L, datasetOriginalSpace);

        Bruteforce_Search<Pixel8Bit> *bruteforceOriginalSpace = new Bruteforce_Search<Pixel8Bit>(imagesOriginalSpace);
        Bruteforce_Search<Pixel16Bit> *bruteforceNewSpace = new Bruteforce_Search<Pixel16Bit>(imagesNewSpace);

        // Open output file
        std::ofstream outputStream(outputFile);
        
        bool repeat;
        do {
            // Read querysets
            Dataset<Pixel8Bit> *querysetOriginalSpace = new Dataset<Pixel8Bit>(queryFileOriginalSpace);
            Dataset<Pixel16Bit> *querysetNewSpace = new Dataset<Pixel16Bit>(queryFileNewSpace);

            if (!querysetOriginalSpace->isValid()) {
                delete querysetOriginalSpace;
                break;
            }

            // Get query images
            std::vector<Image<Pixel8Bit>*> queryImagesOriginalSpace = querysetOriginalSpace->getImages();
            std::vector<Image<Pixel16Bit>*> queryImagesNewSpace = querysetNewSpace->getImages();
            double total_bf_time=0.0,total_lsh_time=0.0,total_reduced_time=0.0,approximation_factor_lsh=0.0,approximation_factor_reduced=0.0;
            unsigned int lsh_found_neighbours = queryImagesOriginalSpace.size();
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
                outputStream << "Nearest neighbor LSH: ";

                double reducedDistanceOriginalSpace;
                if (reducedNearestNeighbours[0].second != exactNearestNeighbours[0].second) {
                    reducedDistanceOriginalSpace = queryImagesOriginalSpace[i]->distance(imagesOriginalSpace[reducedNearestNeighbours[0].second],1);
                } else {
                    reducedDistanceOriginalSpace = exactNearestNeighbours[0].first;
                }

                if (lshNearestNeighbours.size() > 0) {
                    outputStream << lshNearestNeighbours[0].second;
                    approximation_factor_lsh += lshNearestNeighbours[0].first/exactNearestNeighbours[0].first;
                }
                else {
                    outputStream << "NOT FOUND";
                    lsh_found_neighbours--;
                }
                approximation_factor_reduced += reducedDistanceOriginalSpace/exactNearestNeighbours[0].first;
                
                outputStream << std::endl;
                outputStream << "Nearest neighbor True: " << exactNearestNeighbours[0].second << std::endl;
                outputStream << "distanceReduced: " << reducedDistanceOriginalSpace << std::endl;
                outputStream << "distanceLSH: " << (lshNearestNeighbours.size() > 0 ? lshNearestNeighbours[0].first : exactNearestNeighbours[0].first); 
                outputStream << std::endl;
                outputStream << "distanceTrue: " << exactNearestNeighbours[0].first << std::endl << std::endl;
            }

            outputStream << "tReduced: " << total_reduced_time << std::endl;
            outputStream << "tLSH: " << total_lsh_time << std::endl;
            outputStream << "tTrue: " << total_bf_time << std::endl;

            approximation_factor_lsh /= lsh_found_neighbours;
            approximation_factor_reduced /= queryImagesOriginalSpace.size();

            outputStream << "Approximation Factor LSH: " << approximation_factor_lsh << std::endl;
            outputStream << "Approximation Factor Reduced: " << approximation_factor_reduced << std::endl;
            
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
