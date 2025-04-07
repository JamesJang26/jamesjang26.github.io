---
layout: single
title: "Publications"
permalink: /publications/
---
<div class="archive-list">
  {% for pub in site.publications %}
    <article class="post">
      {% if pub.image %}
      <div class="post-thumbnail">
        <a href="{{ pub.link }}" target="_blank">
          <img src="{{ pub.image }}" alt="{{ pub.title }}">
        </a>
      </div>
      {% endif %}
      <div class="post-content">
        <header class="post-header">
          <h2 class="post-title">
            <a href="{{ pub.link }}" target="_blank">{{ pub.title }}</a>
          </h2>
          <p class="post-meta">
            {% if pub.venue %}
              Venue: {{ pub.venue }} &nbsp;|&nbsp;
            {% endif %}
            {% if pub.date %}
              Published on: {{ pub.date | date: "%B %d, %Y" }}
            {% endif %}
          </p>
        </header>
        <div class="post-excerpt">
          {{ pub.description }}
        </div>
        <footer class="post-footer">
          <a href="{{ pub.link }}" target="_blank" class="btn btn-primary">View PDF</a>
        </footer>
      </div>
    </article>
  {% endfor %}
</div>

