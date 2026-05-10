import os
from groq import Groq
from dotenv import load_dotenv
import json
import re

load_dotenv()

client = Groq(api_key="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")


def get_skill_gap_from_groq(student_name, target_role, student_skills):
    print(f"\nCalling Groq API for {target_role}...")

    prompt = f"""
You are a career counselor and skill gap analyst.

A student named {student_name} wants to become a {target_role}.

Their current skills are: {', '.join(student_skills)}

Please analyze and return a JSON response with exactly this structure:
{{
    "required_skills": ["skill1", "skill2", "skill3", "skill4", "skill5", "skill6", "skill7", "skill8", "skill9", "skill10"],
    "matched_skills": ["skills student already has"],
    "gap_skills": [
        {{"skill": "skill name", "priority": "HIGH", "reason": "why this skill is important", "course": "recommended free course"}},
        {{"skill": "skill name", "priority": "MEDIUM", "reason": "why this skill is important", "course": "recommended free course"}},
        {{"skill": "skill name", "priority": "LOW", "reason": "why this skill is important", "course": "recommended free course"}}
    ],
    "market_fit_score": 45,
    "market_fit_label": "Moderate Fit",
    "personalized_advice": "2-3 sentences of personalized career advice for this specific student",
    "learning_roadmap": "Step by step what the student should learn first, second, third"
}}

Rules:
- required_skills should have exactly 10 most important skills for {target_role}
- matched_skills should only include skills from student current skills that match required skills
- gap_skills should be required skills that student does NOT have
- priority HIGH means critical for getting hired, MEDIUM means good to have, LOW means bonus
- market_fit_score is percentage 0-100 based on how many required skills student has
- courses should be free or widely available
- Return ONLY the JSON, no extra text
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a career counselor. Always respond with valid JSON only. No extra text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=2000
    )

    return response.choices[0].message.content


def parse_groq_response(response_text):
    try:
        # Clean response
        response_text = response_text.strip()

        # JSON block extract karo agar extra text ho
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group()

        data = json.loads(response_text)
        return data

    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Raw Response: {response_text}")
        return None


def display_results(data, student_name, target_role):
    print("\n" + "="*60)
    print(f"   SKILL GAP ANALYSIS REPORT")
    print("="*60)
    print(f"   Student     : {student_name}")
    print(f"   Target Role : {target_role}")
    print(f"   Market Fit  : {data['market_fit_score']}% — {data['market_fit_label']}")

    print(f"\n   REQUIRED SKILLS FOR {target_role.upper()}:")
    for skill in data['required_skills']:
        print(f"      → {skill}")

    print(f"\n   MATCHED SKILLS ({len(data['matched_skills'])}):")
    if data['matched_skills']:
        for skill in data['matched_skills']:
            print(f"      ✓ {skill}")
    else:
        print("      None matched")

    print(f"\n   SKILL GAPS WITH PRIORITY:")
    high   = [g for g in data['gap_skills'] if g['priority'] == 'HIGH']
    medium = [g for g in data['gap_skills'] if g['priority'] == 'MEDIUM']
    low    = [g for g in data['gap_skills'] if g['priority'] == 'LOW']

    if high:
        print("\n   HIGH PRIORITY:")
        for gap in high:
            print(f"      [{gap['skill'].upper()}]")
            print(f"      Why: {gap['reason']}")
            print(f"      Course: {gap['course']}")

    if medium:
        print("\n   MEDIUM PRIORITY:")
        for gap in medium:
            print(f"      [{gap['skill'].upper()}]")
            print(f"      Why: {gap['reason']}")
            print(f"      Course: {gap['course']}")

    if low:
        print("\n   LOW PRIORITY:")
        for gap in low:
            print(f"      [{gap['skill'].upper()}]")
            print(f"      Why: {gap['reason']}")
            print(f"      Course: {gap['course']}")

    print(f"\n   PERSONALIZED ADVICE:")
    print(f"   {data['personalized_advice']}")

    print(f"\n   LEARNING ROADMAP:")
    print(f"   {data['learning_roadmap']}")

    print("\n" + "="*60)


def get_student_input():
    print("\nSTUDENT SKILL GAP ANALYSIS")
    print("Powered by Groq LLM")

    student_name  = input("\n   Student Name        : ")
    target_role   = input("   Target Job Role     : ")
    skills_input  = input("   Your Current Skills : ")

    student_skills = [s.strip().lower() for s in skills_input.split(',')]
    student_skills = [s for s in student_skills if s]

    print(f"\n   Name   : {student_name}")
    print(f"   Role   : {target_role}")
    print(f"   Skills : {student_skills}")

    return student_name, target_role, student_skills


def main():
    print("SKILL GAP ANALYSIS USING GROQ LLM")
    print("Dynamic — No Database Required!")

    # Student input
    student_name, target_role, student_skills = get_student_input()

    # Groq API call
    raw_response = get_skill_gap_from_groq(
        student_name, target_role, student_skills
    )

    # Parse response
    data = parse_groq_response(raw_response)

    if data:
        display_results(data, student_name, target_role)
    else:
        print("Error parsing response. Please try again.")

    return data


if __name__ == "__main__":
    main()
