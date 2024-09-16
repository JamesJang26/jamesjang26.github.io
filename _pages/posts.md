---
layout: archive
title: "Posts by Year"
permalink: /posts/
author_profile: true
---

{% for year in site.posts | group_by_exp:"post", "post.date | date: '%Y'" %}
  <h2 id="{{ year.name | slugify }}" class="archive__subtitle">{{ year.name }}</h2>
  <ul class="posts">
    {% for post in year.items %}
      <li>
        <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
        <h3>
          <a class="post-link" href="{{ post.url }}">{{ post.title }}</a>
        </h3>
      </li>
    {% endfor %}
  </ul>
{% endfor %}
