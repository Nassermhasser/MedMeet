import React from 'react'
import { assets } from '../assets/assets'

const About = () => {
  return (
    <div>
      <div className="text-center text-2xl pt-10 text-[#707070]">
        <p>
          ABOUT <span className="text-gray-700 font-semibold">US</span>
        </p>
      </div>

      <div className="my-10 flex flex-col md:flex-row gap-12">
        <img
          className="w-full md:max-w-[360px]"
          src={assets.about_image}
          alt="About MedMeet"
        />
        <div className="flex flex-col justify-center gap-6 md:w-2/4 text-sm text-gray-600">
          <p>
            Welcome to MedMeet — your trusted Moroccan platform for convenient and
            efficient healthcare access. We understand the challenges many people
            face when booking appointments and managing their health, and we’re here
            to simplify it all for you.
          </p>
          <p>
            MedMeet is committed to advancing digital healthcare in Morocco. Our
            platform is constantly improving to provide a smooth, reliable, and
            user-friendly experience. Whether it's your first appointment or
            long-term care, MedMeet is with you every step of the way.
          </p>
          <b className="text-gray-800">Our Vision</b>
          <p>
            Our vision is to make quality healthcare easy to access for everyone. We
            aim to build strong connections between patients and healthcare
            providers, ensuring you get the right care when and where you need it.
          </p>
        </div>
      </div>

      <div className="text-xl my-4">
        <p>
          WHY <span className="text-gray-700 font-semibold">CHOOSE US</span>
        </p>
      </div>

      <div className="flex flex-col md:flex-row mb-20">
        <div className="border px-10 md:px-16 py-8 sm:py-16 flex flex-col gap-5 text-[15px] hover:bg-primary hover:text-white transition-all duration-300 text-gray-600 cursor-pointer">
          <b>EFFICIENCY:</b>
          <p>Effortless appointment scheduling tailored for your busy lifestyle.</p>
        </div>
        <div className="border px-10 md:px-16 py-8 sm:py-16 flex flex-col gap-5 text-[15px] hover:bg-primary hover:text-white transition-all duration-300 text-gray-600 cursor-pointer">
          <b>CONVENIENCE:</b>
          <p>
            Easily connect with trusted doctors and clinics across Morocco — from
            big cities to remote towns.
          </p>
        </div>
        <div className="border px-10 md:px-16 py-8 sm:py-16 flex flex-col gap-5 text-[15px] hover:bg-primary hover:text-white transition-all duration-300 text-gray-600 cursor-pointer">
          <b>PERSONALIZATION:</b>
          <p>
            Receive personalized reminders and recommendations to stay on top of your
            health.
          </p>
        </div>
      </div>
    </div>
  )
}

export default About
