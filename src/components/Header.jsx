import React from 'react'
import { assets } from '../assets/assets'

const Header = () => {
  return (
    <div className='flex flex-col md:flex-row bg-teal-500 text-white rounded-3xl px-6 md:px-10 lg:px-20 py-12 md:py-20 shadow-md overflow-hidden'>

      {/* Left Section */}
      <div className='md:w-1/2 flex flex-col justify-center gap-6'>
        <p className='text-3xl md:text-4xl lg:text-5xl font-semibold leading-tight'>
          Schedule Your Appointment <br /> with Trusted Medical Experts
        </p>
        <div className='flex flex-col md:flex-row items-center gap-4 text-sm font-light'>
          <img className='w-28' src={assets.group_profiles} alt="Group" />
          <p className='text-center md:text-left'>
            Easily explore our network of verified doctors and book your appointment with confidence.
          </p>
        </div>
        <a
          href='#speciality'
          className='flex items-center gap-2 bg-white text-teal-700 px-8 py-3 rounded-full font-medium shadow hover:bg-gray-100 transition'
        >
          Book appointment
          <img className='w-3' src={assets.arrow_icon} alt="Arrow" />
        </a>
      </div>

      {/* Right Section (Image) */}
      <div className='md:w-1/2 flex justify-end items-end mt-10 md:mt-0'>
        <img
          className='w-[200%] max-w-[650px] object-contain'
          src={assets.header_img}
          alt="Header visual"
        />
      </div>
    </div>
  )
}

export default Header
