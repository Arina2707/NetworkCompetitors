# Tool for Bayes Networks creation for laser companies evaluation.

This is the fourth part of **5-stage project**, connected with scoring the companies in laser industry.

Other parts of project:
- https://github.com/jularina/CompetitorsAmpl1
- https://github.com/jularina/CompetitorsAmpl2
- https://github.com/jularina/CompetitorsAmpl3
- https://github.com/jularina/CompetitorsAmpl5

Here the main accent is on providing the structure of Bayes Nets for evaluation of suppliers from several main sides.
The info comes here from the MongoDB database, it is analyzed through 4 Nets and for each object 4 scores are received. 
The posterior probabilities are formed with the help of pairwise comparison method of expert analysis, the priors for Nets are received from collected data in the DataBase.

Structure of 4 Nets are shown in Pic.1, Pic.2, Pic.3, Pic.4. 

![Pic.1](https://user-images.githubusercontent.com/56595596/144024506-02978673-9bb2-45ff-8c09-82f3b4537926.png)

<p align="center">
  Network structure for evaluating a competitor's product
</p>

![Pic.2](https://user-images.githubusercontent.com/56595596/144024945-1b4a5a90-6189-4c03-9987-f91d61e63f05.png)

<p align="center">
  Network structure for evaluating competitor technologies
</p>

![Pic.3](https://user-images.githubusercontent.com/56595596/144025095-b3c24407-3ebf-4937-8b74-98d3ffbd1508.png)

<p align="center">
  Network structure for assessing the organizational values of a competitor.
</p>

![Pic.4](https://user-images.githubusercontent.com/56595596/144025189-cf6c04b6-c3bd-4b1f-ab9a-ff68ea79ef2d.png)

<p align="center">
  Network structure for assessing customer interactions.
</p>
