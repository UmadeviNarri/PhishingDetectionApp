import re
from urllib.parse import urlparse
from googlesearch import search

# has_title
def has_title(soup):
    if soup.title is None:
        return 0
    if len(soup.title.text) > 0:
        return 1
    else:
        return 0


# has_input
def has_input(soup):
    if len(soup.find_all("input")):
        return 1
    else:
        return 0


# has_button
def has_button(soup):
    if len(soup.find_all("button")) > 0:
        return 1
    else:
        return 0


# has_image
def has_image(soup):
    if len(soup.find_all("image")) == 0:
        return 0
    else:
        return 1


# has_submit
def has_submit(soup):
    for button in soup.find_all("input"):
        if button.get("type") == "submit":
            return 1
        else:
            pass
    return 0


# has_link
def has_link(soup):
    if len(soup.find_all("link")) > 0:
        return 1
    else:
        return 0


# has_password
def has_password(soup):
    for input in soup.find_all("input"):
        if (input.get("type") or input.get("name") or input.get("id")) == "password":
            return 1
        else:
            pass
    return 0


# has_email_input
def has_email_input(soup):
    for input in soup.find_all("input"):
        if (input.get("type") or input.get("id") or input.get("name")) == "email":
            return 1
        else:
            pass
    return 0


# has_hidden_element
def has_hidden_element(soup):
    for input in soup.find_all("input"):
        if input.get("type") == "hidden":
            return 1
        else:
            pass
    return 0


# has_audio
def has_audio(soup):
    if len(soup.find_all("audio")) > 0:
        return 1
    else:
        return 0


# has_video
def has_video(soup):
    if len(soup.find_all("video")) > 0:
        return 1
    else:
        return 0


# number_of_inputs
def number_of_inputs(soup):
    return len(soup.find_all("input"))


# number_of_buttons
def number_of_buttons(soup):
    return len(soup.find_all("button"))


# number_of_images
def number_of_images(soup):
    image_tags = len(soup.find_all("image"))
    count = 0
    for meta in soup.find_all("meta"):
        if meta.get("type") or meta.get("name") == "image":
            count += 1
    return image_tags + count


# number_of_option
def number_of_option(soup):
    return len(soup.find_all("option"))


# number_of_list
def number_of_list(soup):
    return len(soup.find_all("li"))


# number_of_TH
def number_of_TH(soup):
    return len(soup.find_all("th"))


# number_of_TR
def number_of_TR(soup):
    return len(soup.find_all("tr"))


# number_of_href
def number_of_href(soup):
    count = 0
    for link in soup.find_all("link"):
        if link.get("href"):
            count += 1
    return count


# number_of_paragraph
def number_of_paragraph(soup):
    return len(soup.find_all("p"))


# number_of_script
def number_of_script(soup):
    return len(soup.find_all("script"))


# length_of_title
def length_of_title(soup):
    if soup.title == None:
        return 0
    return len(soup.title.text)


"""
print("has_title --> ", has_title(soup))
print("has_input --> ", has_input(soup))
print("has_button --> ", has_button(soup))
print("has_image --> ", has_image(soup))
print("has_submit --> ", has_submit(soup))
print("has_link --> ", has_link(soup))
print("has_password --> ", has_password(soup))
print("has_email_input --> ", has_email_input(soup))
print("has_hidden_element --> ", has_hidden_element(soup))
print("has_audio --> ", has_audio(soup))
print("has_video --> ", has_video(soup))
print("number_of_inputs --> ", number_of_inputs(soup))
print("number_of_buttons --> ", number_of_buttons(soup))
print("number_of_images --> ", number_of_images(soup))
print("number_of_option --> ", number_of_option(soup))
print("number_of_list --> ", number_of_list(soup))
print("number_of_TH --> ", number_of_TH(soup))
print("number_of_TR --> ", number_of_TR(soup))
print("number_of_href --> ", number_of_href(soup))
print("number_of_paragraph --> ", number_of_paragraph(soup))
print("number_of_script --> ", number_of_script(soup))
print("length_of_title --> ", length_of_title(soup))

"""


# has h1
def has_h1(soup):
    if len(soup.find_all("h1")) > 0:
        return 1
    else:
        return 0


# has h2
def has_h2(soup):
    if len(soup.find_all("h2")) > 0:
        return 1
    else:
        return 0


# has h3
def has_h3(soup):
    if len(soup.find_all("h3")) > 0:
        return 1
    else:
        return 0


# length of text
def length_of_text(soup):
    return len(soup.get_text())


# number of clickable button
def number_of_clickable_button(soup):
    count = 0
    for button in soup.find_all("button"):
        if button.get("type") == "button":
            count += 1
    return count


# number of a
def number_of_a(soup):
    return len(soup.find_all("a"))


# number of img
def number_of_img(soup):
    return len(soup.find_all("img"))


# number of div class
def number_of_div(soup):
    return len(soup.find_all("div"))


# number of figures
def number_of_figure(soup):
    return len(soup.find_all("figure"))


# has footer
def has_footer(soup):
    if len(soup.find_all("footer")) > 0:
        return 1
    else:
        return 0


# has form
def has_form(soup):
    if len(soup.find_all("form")) > 0:
        return 1
    else:
        return 0


# has textarea
def has_text_area(soup):
    if len(soup.find_all("textarea")) > 0:
        return 1
    else:
        return 0


# has iframe
def has_iframe(soup):
    if len(soup.find_all("iframe")) > 0:
        return 1
    else:
        return 0


# has text input
def has_text_input(soup):
    for input in soup.find_all("input"):
        if input.get("type") == "text":
            return 1
    return 0


# number of meta
def number_of_meta(soup):
    return len(soup.find_all("meta"))


# has nav
def has_nav(soup):
    if len(soup.find_all("nav")) > 0:
        return 1
    else:
        return 0


# has object
def has_object(soup):
    if len(soup.find_all("object")) > 0:
        return 1
    else:
        return 0


# has picture
def has_picture(soup):
    if len(soup.find_all("picture")) > 0:
        return 1
    else:
        return 0


# number of sources
def number_of_sources(soup):
    return len(soup.find_all("source"))


# number of span
def number_of_span(soup):
    return len(soup.find_all("span"))


# number of table
def number_of_table(soup):
    return len(soup.find_all("table"))


# adding url features to content features

# Use of IP or not in domain
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'  # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0



def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0


# pip install googlesearch-python


def google_index(url):
    site = search(url, 5)
    return 1 if site else 0



def count_dot(url):
    count_dot = url.count('.')
    return count_dot



def count_www(url):
    url.count('www')
    return url.count('www')



def count_atrate(url):
    return url.count('@')



def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')



def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')



def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0



def count_https(url):
    return url.count('https')



def count_http(url):
    return url.count('http')



def count_per(url):
    return url.count('%')


def count_ques(url):
    return url.count('?')



def count_hyphen(url):
    return url.count('-')



def count_equal(url):
    return url.count('=')



def url_length(url):
    return len(str(url))




# Hostname Length

def hostname_length(url):
    return len(urlparse(url).netloc)




def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0



def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits




def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters




# First Directory Length
def fd_length(url):
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0






