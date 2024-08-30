import requests
from bs4 import BeautifulSoup
import features as fe


def create_vector_url(url):
            url_features = [
                fe.having_ip_address(url),
                fe.abnormal_url(url),
                fe.google_index(url),
                fe.count_dot(url),
                fe.count_www(url),
                fe.count_atrate(url),
                fe.no_of_dir(url),
                fe.no_of_embed(url),
                fe.shortening_service(url),
                fe.count_http(url),
                fe.count_https(url),
                fe.count_per(url),
                fe.count_ques(url),
                fe.count_hyphen(url),
                fe.count_equal(url),
                fe.url_length(url),
                fe.hostname_length(url),
                fe.suspicious_words(url),
                fe.digit_count(url),
                fe.letter_count(url),
                fe.fd_length(url)
            ]

            return url_features

def create_vector_soup(soup):
    content_features = [
                fe.has_title(soup),
                fe.has_input(soup),
                fe.has_button(soup),
                fe.has_image(soup),
                fe.has_submit(soup),
                fe.has_link(soup),
                fe.has_password(soup),
                fe.has_email_input(soup),
                fe.has_hidden_element(soup),
                fe.has_audio(soup),
                fe.has_video(soup),
                fe.number_of_inputs(soup),
                fe.number_of_buttons(soup),
                fe.number_of_images(soup),
                fe.number_of_option(soup),
                fe.number_of_list(soup),
                fe.number_of_TH(soup),
                fe.number_of_TR(soup),
                fe.number_of_href(soup),
                fe.number_of_paragraph(soup),
                fe.number_of_script(soup),
                fe.length_of_title(soup),
                fe.has_h1(soup),
                fe.has_h2(soup),
                fe.has_h3(soup),
                fe.length_of_text(soup),
                fe.number_of_clickable_button(soup),
                fe.number_of_a(soup),
                fe.number_of_img(soup),
                fe.number_of_div(soup),
                fe.number_of_figure(soup),
                fe.has_footer(soup),
                fe.has_form(soup),
                fe.has_text_area(soup),
                fe.has_iframe(soup),
                fe.has_text_input(soup),
                fe.number_of_meta(soup),
                fe.has_nav(soup),
                fe.has_object(soup),
                fe.has_picture(soup),
                fe.number_of_sources(soup),
                fe.number_of_span(soup),
                fe.number_of_table(soup)
            ]
    return content_features
