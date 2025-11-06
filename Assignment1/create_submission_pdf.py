#!/usr/bin/env python3
"""
Script to create submission PDF
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def create_submission_pdf():
    filename = "submission.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Starting position
    y_position = height - 1.5 * inch

    # Set font
    c.setFont("Helvetica-Bold", 14)

    # Student 1 name and ID
    c.drawString(1 * inch, y_position, "Barak Ozmo - 206897290")
    y_position -= 0.4 * inch

    # Student 2 name and ID
    c.drawString(1 * inch, y_position, "Eilon Udi - 205979800")
    y_position -= 0.6 * inch

    # GitHub link
    c.setFont("Helvetica", 12)
    github_link = "https://github.com/eilonudi-work/MultiAgentCourse"
    c.drawString(1 * inch, y_position, github_link)
    y_position -= 0.6 * inch

    # Score
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, y_position, "Score: 93/100")

    # Finish first page and add a blank second page
    c.showPage()

    # Second page (blank) - just call showPage to finalize it
    c.showPage()

    c.save()
    print(f"✅ PDF created successfully: {filename} (2 pages)")

if __name__ == "__main__":
    create_submission_pdf()
