---
layout: publications
title: "Publications"
permalink: /publications/
author_profile: true
---

{% assign sorted_pubs = site.publications | sort: "date" | reverse %}
<div class="publication-list">
  {% for pub in sorted_pubs %}
    <div class="publication-item">
      <div class="publication-image">
        {% if pub.image %}
          <img src="{{ pub.image | relative_url }}" alt="{{ pub.title }}">
        {% endif %}
      </div>
      <div class="publication-info">
        <h3 class="publication-title">{{ pub.title }}</h3>
        {% if pub.authors %}
          <p class="publication-authors">
            {% for author in pub.authors %}
              {% if author == "Dongsuk Jang" %}
                <strong>{{ author }}</strong>
              {% else %}
                {{ author }}
              {% endif %}
              {% unless forloop.last %}, {% endunless %}
            {% endfor %}
          </p>
        {% endif %}
        {% if pub.venue %}
          <p class="publication-venue"><em>{{ pub.venue }}</em></p>
        {% endif %}
        {% if pub.date %}
          <p class="publication-date">Published on: {{ pub.date | date: "%B %d, %Y" }}</p>
        {% endif %}
        {% if pub.description %}
          <p class="publication-description">{{ pub.description }}</p>
        {% endif %}
        {% if pub.link %}
          <p class="publication-link"><a href="{{ pub.link }}" target="_blank">PDF</a></p>
        {% endif %}
      </div>
    </div>
  {% endfor %}
</div>
