## Prerequisites

Before running the project, ensure that the following tools are installed on your system:

- **Java 8**: Download it from the [official Oracle website](https://www.oracle.com/java/technologies/javase-jdk8-downloads.html).
- **Scala 2.13.12**: Follow the instructions on the [official Scala website](https://www.scala-lang.org/download/).
- **SBT (Scala Build Tool)**: Installation instructions can be found on the [SBT official website](https://www.scala-sbt.org/download.html).
- **Hadoop (for running locally)**: Install Hadoop and follow a basic configuration tutorial available online.

## Running the Code

### Steps to Run with SBT
Start the SBT shell by typing: sbt -> clean -> reload -> update -> compile -> run

### Testing the code
Start the SBT shell by typing: sbt -> test

### Run code on local with Hadoop
1. Make sure to set up Hadoop correctly on your local.
2. Create a jar of your program. You can find [tutorial](https://www.baeldung.com/scala/sbt-fat-jar) to set up and create a jar.
3. Run the Hadoop system. Type $ jps to check to see if everything is running. If everything is correctly set up, the result will list all the corresponding process IDs for the four main components of the Hadoop ecosystem: NameNode, DataNode, NodeManager, and ResourceManager. 
4. Set up an input folder on your Hadoop system where all the input files are located.
5. Then run this command: $ hadoop jar [path_to_jar_file] [name_of_drive_class] [path_to_input_folder_on_Hadoop] [path_you_want_Hadoop_to_store_output]
6. After running, Hadoop will return a summary of the MapReduce execution. You can base on this to see if the output is created.

### Some tips to debug
The program integrates Logback with SL4J to log all the necessary information during the execution process. You can find these logs via on the Hadoop UI, or navigate to the userlogs folder on your local's filesystem. Remember to choose the correct folder of the application process you want to check. Each application process logs the process, and you can check it in the folder syslog..
