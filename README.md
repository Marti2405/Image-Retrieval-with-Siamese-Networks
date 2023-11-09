# Image Retrieval with Siamese Networks

This project aims to explore the effectiveness of a **Siamese Neural Network** (SNN) in developing a robust **image retrieval** model by leveraging **image key points**. Through rigorous experimentation and analysis, the project investigates the SNN's capabilities in capturing intricate image similarities and addresses the challenges associated with image retrieval tasks. The achieved performance demonstrates the SNN's potential in enhancing image retrieval accuracy and robustness.

## Research Paper
The comprehensive research paper detailing the methodology, experiments, and findings of this project is included in the repository. Please refer to the [Research Paper](./ResearchPaper.pdf) for an in-depth analysis of the project, including detailed descriptions of the methodologies employed, experimental setups, results, and discussions.

## Method
The project employs a Siamese Neural Network architecture, incorporating image key points for feature extraction and similarity measurement. The network undergoes comprehensive training using a dataset containing image pairs and corresponding relevance scores. Evaluation metrics such as precision at k and accuracy within a relevance score range are utilized to assess the model's performance.

Image Key-point Extraction:
![SNN](Images/ORB.png)

Our Siamese Neural Network Model:
![SNN](Images/SNN.png)

## Results

- The Siamese Neural Network (SNN) achieved a 10-precision of 2%.
- The SNN demonstrated an accuracy of 85.07% in classifying relevant images within a relevance score range of 0.01.
- Manual inspection revealed that between 10 and 50% of the top 10 retrieved images were relevant, while the rest exhibited similar visual features.

## Conclusion

The results highlight the SNN's effectiveness in accurately identifying relevant images, demonstrating its potential for practical image retrieval applications. However, the model's performance indicates the need for more sophisticated feature extraction techniques to handle nuanced image similarities. Future improvements may involve the integration of advanced feature descriptors and convolutional layers to enhance the model's performance in complex image retrieval scenarios.

## Repository Files
- `model.ipynb`: Python notebook script containing the implementation of the Siamese Neural Network and the image retrieval process.
- `process_data.ipynb`: Python notebook script for preprocessing the dataset and extracting image key points.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
