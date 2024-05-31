<p align="center">
<img src="https://github.com/Syntax-Savior/Credit-Card-Fraud-Detection_System/blob/main/Assets/Images/St_Marys-Logo-Resized.jpg">
</p>

<h1 align="center">St. Mary's University</h1>

<h2 align="center">Department of Computer Science</h2>
<h3 align="center">Selected Topics in Computer Science<h3>

                  Team Members

|        Name         |          Id       |
|---------------------|-------------------|
| Alemayehu Kibret    |     RCD/0544/2013 |
| Mesoud Mohammed     |     RCD/3070/2013 |
| Simon Michael       |     RCD/0573/2013 |
| Naod Akono          |     RCD/0569/2013 |

<hr style="height:2px; margin-bottom:25px;" >

<h2 align="center">Fraud Detection System</h2>

<p>
  <h1 style="margin-bottom:10px">Chapter 1</h1>
  <h2 style="margin-bottom:10px">Introduction</h2>
  <h3 style="margin-bottom:5px">The problem of Fraud Detection</h3>

  <p>
  Fraud is as old as humanity itself and can take an unlimited variety of different forms. Moreover, the development of new technologies provides additional ways in which criminals may commit fraud, for instance in e-commerce the information about the card is sufficient to perpetrate a fraud. Financial losses due to fraud affect not only merchants and banks (e.g. reimbursements), but also individual clients. If the bank loses money, customers eventually pay as well through higher interest rates, higher membership fees, etc. Fraud may also affect the reputation and image of a merchant causing non-financial losses that, though difficult to quantify in the short term, may become visible in the long period. For example, if a cardholder is victim of fraud with a certain company, he may no longer trust their business and choose a competitor.
  </p>

  <p>
  The actions taken against fraud can be divided into fraud prevention, which attempts to block fraudulent transactions at source, and fraud detection, where successful fraud transactions are identified a posterior. Technologies that have been used in order to prevent fraud are Address Verification Systems (AVS), Card Verification Method (CVM) and Personal Identification Number (PIN). AVS involves verification of the address with zip code of the customer while CVM and PIN involve checking of the numeric code that is keyed in by the customer. For prevention purposes, financial institutions challenge all transactions with rule based filters and data mining methods as neural networks.
  </p>

  <p>
  Fraud detection is, given a set of credit card transactions, the process of identifying if a new authorized transaction belongs to the class of fraudulent or genuine transactions. A Fraud Detection System (FDS) should not only detect fraud cases efficiently, but also be cost-effective in the sense that the cost invested in transaction screening should not be higher than the loss due to frauds. Tej Paul Bhatla, the SVP and business head at TCS shows that screening only 2% of transactions can result in reducing fraud losses accounting for 1% of the total value of transactions. However, a review of 30% of transactions could reduce the fraud losses drastically to 0.06%, but increase the costs exorbitantly. In order to minimize costs of detection it is important to use expert rules and statistical based models (e.g. Machine Learning) to make a first screen between genuine and potential fraud and ask the investigators to review only the cases with high risk.
  </p>

 <p>
  This project focuses on different types of frauds mainly credit card fraud, Phishing and Social Engineering, identity theft, E-commerce Fraud, and Data Breaches. We combined the different types of fraud detection methods like rule-based systems, anomaly detection, and machine learning models and created a <strong>hybrid fraud detection system</strong>. Now let's briefly look at each types of frauds that we mentioned above.

  * **_Credit Card Fraud_**: is the unauthorized use of an individual’s credit card or card information to make purchases, or to remove funds from the cardholder’s account.

  * **_Phishing and Social Engineering_**: Phishing and social engineering are methods of deceiving people into revealing sensitive information or performing malicious actions. Phishing is a form of social engineering that uses email or malicious websites to impersonate a trustworthy organization. Social engineering can also use other techniques, such as phone calls, to psychologically manipulate people into divulging information or taking inappropriate actions.

  * **_Identity Theft_**: is an instance where someone uses personal or financial information which belongs to someone and without their permission. Criminals might steal names and addresses, credit card, or bank account numbers, social security number, or medical insurance account numbers. They could use them to make unauthorized purchases, get new credit cards in the victim's name, gain access to their accounts or personal information, etc.

  * **_E-Commerce Fraud_**: refers to various criminal activities that occur within the online shopping and transaction environment. A few types of e-commerce frauds can be chargeback fraud, account takeover fraud, refund fraud, affiliate fraud, counterfeit or fake products, drop-shipping fraud, etc.

  * **_Data Breaches_**: is a security incident in which malicious insiders or external attackers gain unauthorized access to confidential data or sensitive information such as medical records, financial information, or personally identifiable information (PII). These breaches can have serious consequences for organizations and individuals alike.

  <h4>Implementation Steps of our Hybrid Model:</h4>

  1. **_Data Collection_**: This is the first step of many where we collected data from various sources. Specifically, from Kaggle, Cornell University's senior papers, arxiv, and datacamp. Researchers have demonstrated that massive data can lead to lower estimation variance and hence better predictive performance. Therefore, we have fed each of our models approximately over 3 million data entries.

  2. **_Data Pre-processing_**: We imported python modules like pandas and scikit-learn and cleaned and pre-processed our data, properly handled missing values, encoded categorical variables, and when dealing with unbalanced datasets we created our own model of a balanced dataset.

  3. **_Data Analysis_**: The data analysis phase involved exploring the dataset using descriptive statistics and visualizations (for example using histograms, boxplots, scatter plots, etc). It also includes pattern identification that have significant differences between fraudulent and legitimate transactions.

  4. **_Data Splitting_**: The datasets were split into training sets (80%) and testing sets (20%).

  5. **_Rule-Based System_**: Implemented predefined rules in our code to flag obvious cases of fraud.

  6. **_Anomaly Detection_**: Applied anomaly detection techniques such as isolation forest algorithm to identify unusual patterns in the data.

  7. **_Machine Learning Models_**: Trained machine learning models on historical data to detect more complex fraud patterns.

  8. **_Integration_**: Combined the outputs from all three methods to make a final decision.

  9. **_Evaluation_**: Evaluated the performance of our hybrid system using appropriate metrics and tested it on unseen data to ensure its effectiveness.

  <p>
  <h3>Model Selection</h3>
  After experimenting with different models we have seen that every algorithm that we have used (Logistic Regression, Random Forest, Isolation Forest, Gradient Boosting, Support Vector Machines (SVM), Neural Networks) excels in different aspects. Due to the complexity of the task at hand we decided to use multiple models and combine their predictions to yield better results. Therefore, we decided to go with an approach called Ensemble Learning.

  Ensemble Learning is the process of combining the predictions of multiple models using ensemble techniques. Here are a few common ensemble methods that we have used:

  * **_Voting_**: Combines predictions by majority voting (for classification tasks) or averaging (for
            regression tasks).
  * **_Stacking_**: Train a meta-model on the predictions of base models. The meta-model learns how to best
              combine the base models' predictions.
  * **_Bagging_**: Build multiple models independently and combine their predictions. Random forests are an
            example of a bagging technique.
  * **_Boosting_**: Build models sequentially, with each new model correcting errors made by the previous
              ones. Gradient boosting is a popular boosting technique.

  Finally, we trained our model using the above algorithms and used techniques like hyperparameter tuning to optimize the model's performance.
  </p>
 </p>
</p>

<h3 align="center">Credit Card Fraud Detection System</h3>

<p align="center">
<img src="https://github.com/Syntax-Savior/Credit-Card-Fraud-Detection_System/blob/main/Assets/Images/WorkFlow-CCFDS.jpg">
</p>