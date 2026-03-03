# RES-Q AI

## Overview
RES-Q AI is an intelligent disaster detection and emergency analysis system that combines traditional Machine Learning with Large Language Models to identify real disaster events from SMS or social media messages in real time.

The system first uses a TF-IDF + Logistic Regression classifier to determine whether a message represents a genuine disaster. If classified as high-risk, it triggers Gemini AI to perform advanced reasoning — including severity assessment, live news retrieval, safety guidance, and emergency action planning.

RES-Q AI is designed to work in low-connectivity environments and can be integrated with SMS gateways, emergency dashboards, or disaster management systems.

## Problem Statement
During natural disasters and emergency situations:

Internet connectivity may be unreliable.

Emergency services receive overwhelming and unverified reports.

False alarms and misinformation spread rapidly.

There is no intelligent filtering mechanism to prioritize real threats.

Critical emergency messages are often delayed or ignored.

Communities and emergency authorities require a scalable, intelligent system that can:

Automatically detect genuine disaster reports.

Classify the severity of the situation.

Fetch real-time contextual information.

Provide actionable emergency response recommendations.

Without such automation, response times increase and resource allocation becomes inefficient.
## Solution Architecture
RES-Q AI follows a hybrid AI architecture that balances speed, cost-efficiency, and intelligence.

Step-by-Step Flow:

User Input (SMS / Tweet / Text Message)

Text Preprocessing

Lowercasing

Removing URLs, mentions, numbers, punctuation

TF-IDF Vectorization

Logistic Regression Classification

Determines if message is a real disaster

Computes probability score

Threshold-Based Decision

If probability exceeds threshold → Trigger Gemini

Gemini AI Analysis

Disaster type classification

Severity level detection

Live news retrieval using Google Search tool

Safety cautions

Emergency action plan

Location inference (if mentioned)

Structured Emergency Output

This hybrid approach ensures:

Fast initial filtering using traditional ML.

Advanced reasoning only when necessary.

Reduced API cost and improved scalability.
## Tech Stack
Programming & Backend

Python

Flask (API integration)

Machine Learning

Scikit-learn

TF-IDF Vectorizer

Logistic Regression (with class balancing)

Train-test split validation

Data Processing

Pandas

Regular Expressions (Text Cleaning)

AI & Intelligence Layer

Google Gemini API

Google Search Tool Integration (Live News Retrieval)

Evaluation Metrics

Accuracy

F1 Score

Confusion Matrix

