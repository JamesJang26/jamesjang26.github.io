---
layout: single
title: "Publications"
permalink: /publications/
---
{% for pub in site.publications %}
  <div class="publication-item" style="margin-bottom: 2rem;">
    <h2>{{ pub.title }}</h2>
    {% if pub.image %}
      <img src="{{ pub.image }}" alt="{{ pub.title }}" style="max-width:200px;">
    {% endif %}
    <p>{{ pub.description }}</p>
    {% if pub.link %}
      <a href="{{ pub.link }}" target="_blank">View Paper</a>
    {% endif %}
  </div>
{% endfor %}
