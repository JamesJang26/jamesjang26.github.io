---
layout: single
title: "Publications"
permalink: /publications/
author_profile: true
---

<div class="publication-list">
  {% for pub in site.publications %}
  <div class="publication-item" style="display: flex; align-items: flex-start; margin-bottom: 2rem;">
    <!-- 왼쪽 이미지 -->
    <div style="flex: 0 0 auto; margin-right: 20px;">
      {% if pub.image %}
        <img src="{{ pub.image | relative_url }}" alt="{{ pub.title }}" style="width: 300px; max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px">
      {% endif %}
    </div>

    <!-- 오른쪽 텍스트 -->
    <div style="flex: 1;">
      <h3 style="margin-top: 0;">{{ pub.title }}</h3>
      {% if pub.authors %}
        <p>
          {% for author in pub.authors %}
            {% if author == "James Jang" %}
              <strong>{{ author }}</strong>
            {% else %}
              {{ author }}
            {% endif %}
            {% unless forloop.last %}, {% endunless %}
          {% endfor %}
        </p>
      {% endif %}

      {% if pub.venue %}
        <p><em>{{ pub.venue }}</em></p>
      {% endif %}
      {% if pub.date %}
        <p>Published on: {{ pub.date | date: "%B %d, %Y" }}</p>
      {% endif %}
      {% if pub.description %}
        <p>{{ pub.description }}</p>
      {% endif %}
      {% if pub.link %}
        <p><a href="{{ pub.link }}" target="_blank">PDF</a></p>
      {% endif %}
    </div>
  </div>
  {% endfor %}
</div>
