---
layout: default
title: "Publications"
permalink: /publications/
---
<h1>{{ page.title }}</h1>
<div>
  {% for pub in site.publications %}
    <div>
      <h2>{{ pub.title }}</h2>
      <img src="{{ pub.image }}" alt="{{ pub.title }}" style="max-width:200px;">
      <p>{{ pub.description }}</p>
      <a href="{{ pub.link }}" target="_blank">View Paper</a>
    </div>
  {% endfor %}
</div>
