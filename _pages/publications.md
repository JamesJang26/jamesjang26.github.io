---
layout: single
title: "Publications"
permalink: /publications/
author_profile: true
---

<div class="publications-list">
  {% for pub in site.publications %}
  <div class="publication-item" style="display: flex; align-items: flex-start; margin-bottom: 2rem;">
    <!-- 왼쪽 이미지 영역 -->
    <div style="margin-right: 20px;">
      {% if pub.image %}
        <img src="{{ pub.image }}" alt="{{ pub.title }}" style="max-width: 120px;">
      {% endif %}
    </div>

    <!-- 오른쪽 텍스트 영역 -->
    <div>
      <!-- 논문 제목 -->
      <h3 style="margin-top: 0;">{{ pub.title }}</h3>

      <!-- 저자 목록 (내 이름만 굵게 표시 예시) -->
      {% if pub.authors %}
        <p>
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

      <!-- venue(학회/저널명) -->
      {% if pub.venue %}
        <p><em>{{ pub.venue }}</em></p>
      {% endif %}

      <!-- 출판 날짜 -->
      {% if pub.date %}
        <p>Published on: {{ pub.date | date: "%B %d, %Y" }}</p>
      {% endif %}

      <!-- PDF 링크 / BibTeX 등 -->
      {% if pub.link %}
        <p><a href="{{ pub.link }}" target="_blank">PDF / BibTeX</a></p>
      {% endif %}

      <!-- 간단 설명 (description) -->
      {% if pub.description %}
        <p>{{ pub.description }}</p>
      {% endif %}
    </div>
  </div>
  {% endfor %}
</div>
