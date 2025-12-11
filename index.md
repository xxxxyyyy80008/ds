---
layout: default
title: Data Science
nav_order: 1
---

# Data Science

This section covers the full data science workflow:

- Data processing
- Feature engineering
- Model selection and evaluation
- End-to-end workflows and best practices

Use the left sidebar to navigate, or start with the topics below.

## Contents

<ul>
  {% assign docs_pages = site.pages
       | where_exp: "p", "p.path contains 'docs/'"
       | sort: "nav_order" %}
  {% for p in docs_pages %}
    {% if p.title and p.nav_exclude != true %}
      <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>






## Topics

- [Data Processing](./data-processing/)
- [Feature Engineering](./feature-engineering/)
- [Modeling & Evaluation](./modeling/)
- [Workflows & Best Practices](./workflows/)
