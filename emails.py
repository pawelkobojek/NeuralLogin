def get_emails(emails_file="emails.txt"):
    with open(emails_file) as f:
        return filter(lambda x: x[0] != '#', f.read().splitlines())
