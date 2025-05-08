import React from 'react'
import { assets } from '../assets/assets'

const About = () => {
  return (
    <div>

      <div className='text-center text-2xl pt-10 text-[#707070]'>
        <p>ABOUT <span className='text-gray-700 font-semibold'>US</span></p>
      </div>

      <div className='my-10 flex flex-col md:flex-row gap-12'>
        <img className='w-full md:max-w-[360px]' src={assets.about_image} alt="" />
        <div className='flex flex-col justify-center gap-6 md:w-2/4 text-sm text-gray-600'>
          <p>Welcome to MedMeet, your trusted platform for convenient and efficient healthcare management. At MedMeet, we recognize the challenges individuals face when booking doctor appointments and managing their health records, and we are here to simplify the process for you.</p>
          <p>MedMeet is dedicated to advancing healthcare technology, constantly improving our platform to deliver a seamless user experience. Whether you're scheduling your first appointment or managing ongoing care, MedMeet is your reliable partner in every step of your healthcare journey.</p>
          <b className='text-gray-800'>Our Vision</b>
          <p>At MedMeet, our vision is to provide a seamless healthcare experience for all users. We strive to connect patients with healthcare providers easily and efficiently, ensuring you receive the care you need, whenever you need it.</p>
        </div>
      </div>

      <div className='text-xl my-4'>
        <p>WHY  <span className='text-gray-700 font-semibold'>CHOOSE US</span></p>
      </div>

      <div className='flex flex-col md:flex-row mb-20'>
        <div className='border px-10 md:px-16 py-8 sm:py-16 flex flex-col gap-5 text-[15px] hover:bg-primary hover:text-white transition-all duration-300 text-gray-600 cursor-pointer'>
          <b>EFFICIENCY:</b>
          <p>Effortless appointment scheduling designed to fit seamlessly into your busy lifestyle.</p>
        </div>
        <div className='border px-10 md:px-16 py-8 sm:py-16 flex flex-col gap-5 text-[15px] hover:bg-primary hover:text-white transition-all duration-300 text-gray-600 cursor-pointer'>
          <b>CONVENIENCE: </b>
          <p>Connect with a network of trusted healthcare professionals right in your area.</p>
        </div>
        <div className='border px-10 md:px-16 py-8 sm:py-16 flex flex-col gap-5 text-[15px] hover:bg-primary hover:text-white transition-all duration-300 text-gray-600 cursor-pointer'>
          <b>PERSONALIZATION:</b>
          <p >Personalized recommendations and timely reminders to keep you proactive about your health.</p>
        </div>
      </div>

    </div>
  )
}

export default About
