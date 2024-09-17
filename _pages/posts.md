---
title: "Posts by Year"
permalink: /posts/
layout: archive
author_profile: true
---

{% assign postsByYear = site.posts | group_by_exp: "post", "post.date | date: '%Y'" %}
{% for year in postsByYear %}
  <h2 id="{{ year.name | slugify }}" class="archive__subtitle">{{ year.name }} ({{ year.items | size }} posts)</h2> 
  <ul class="posts-list">
    {% for post in year.items %}
      <li class="post-item">
        <h3 class="post-title">
          <a href="{{ post.url }}">{{ post.title }}</a>
        </h3>
        <p class="post-date">{{ post.date | date: "%b %-d, %Y" }}</p>
      </li>
    {% endfor %}
  </ul>
{% endfor %}
