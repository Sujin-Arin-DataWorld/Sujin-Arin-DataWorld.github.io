---
layout: page
title: About Me
permalink: /about-me/
weight: 1
---

## About Me

<div class="row justify-content-between">
  <div class="col-md-8">
    <p>Hi, I am <strong>{{ site.author.name }}</strong> 👋</p>
    <p>I am a passionate <strong>data analyst</strong> with a strong focus on <strong>finance, machine learning, and data visualization</strong>. My mission is to transform complex data into actionable insights that drive strategic business decisions.</p>
    <p>With expertise in <strong>SQL, Python, Power BI, and Machine Learning</strong>, I specialize in building efficient data models and interactive dashboards that uncover valuable trends and patterns.</p>
  </div>
  <div class="col-md-4 mt-md-0 mt-4">
    <img src="{{ site.author.image }}" alt="{{ site.author.name }}" class="img-fluid rounded-circle">
  </div>
</div>

## Focus Areas

<div class="row">
  <div class="col-md-6">
    <div class="card mb-3">
      <div class="card-body">
        <h5>📊 Financial Data Analysis</h5>
        <p>Uncovering market trends, assessing risks, and generating actionable investment insights.</p>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card mb-3">
      <div class="card-body">
        <h5>🤖 Machine Learning</h5>
        <p>Building predictive models to optimize business strategies and forecast future trends.</p>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card mb-3">
      <div class="card-body">
        <h5>📈 Data Visualization</h5>
        <p>Creating compelling and interactive dashboards for better decision-making.</p>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card mb-3">
      <div class="card-body">
        <h5>💼 Business Intelligence</h5>
        <p>Leveraging data to drive operational improvements and sustainable business growth.</p>
      </div>
    </div>
  </div>
</div>

## Current Projects

<div class="alert alert-info" role="alert">
  <p>I'm currently working on an <strong>NLP-based Financial Sentiment Analysis project</strong> that analyzes economic news and Twitter data to predict stock market trends. In this complex global economic and political landscape, I'm leveraging Python, Machine Learning algorithms, and various APIs to extract valuable insights from text data and correlate sentiment patterns with market movements. 📈</p>
  <p>This challenging project combines my interests in finance with data science techniques including:</p>
  <ul>
    <li>Natural Language Processing for sentiment extraction</li>
    <li>API integration for real-time data collection</li>
    <li>Machine Learning models for trend prediction</li>
    <li>Data visualization for insight communication</li>
  </ul>
  <p>Check out my <a href="../projects/" class="alert-link">Projects page</a> to see my latest work!</p>
</div>

## Skills

{% include about/skills.html title="Technical Skills" source=site.data.programming-skills %}

{% include about/skills.html title="Other Skills" source=site.data.other-skills %}

## Experience & Education

{% include about/timeline.html %}

## Let's Connect!

<div class="row">
  <div class="col-md-8">
    <p>I'm always open to discussing new projects, data challenges, or opportunities to collaborate.</p>
    <p>Feel free to reach out via email or connect with me on LinkedIn and GitHub.</p>
    <p>
      <a class="btn btn-primary" href="mailto:{{ site.author.email }}">
        <i class="fas fa-envelope"></i> Email Me
      </a>
      <a class="btn btn-dark" href="https://github.com/{{ site.author.github }}">
        <i class="fab fa-github"></i> GitHub
      </a>
      {% if site.author.linkedin %}
      <a class="btn btn-info" href="https://www.linkedin.com/in/{{ site.author.linkedin }}">
        <i class="fab fa-linkedin"></i> LinkedIn
      </a>
      {% endif %}
    </p>
  </div>
</div>
