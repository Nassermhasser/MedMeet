import React from 'react'
import { assets } from '../assets/assets'

const Footer = () => {
  return (
    <div className='md:mx-10'>
      <div className='flex flex-col sm:grid grid-cols-[3fr_1fr_1fr] gap-14 my-10  mt-40 text-sm'>

        <div>
          <img className='mb-5 w-40' src={assets.logo} alt="" />
          <p className='w-full md:w-2/3 text-gray-600 leading-6'>MedMeet is your trusted Moroccan platform for booking doctor appointments with ease and confidence. From major cities to remote regions, MedMeet connects patients with top-rated medical professionals across the country. Our simple and modern system makes it easy for everyone to schedule medical consultations, anytime and from anywhere in the country.</p>
        </div>

        <div>
          <p className='text-xl font-medium mb-5'>COMPANY</p>
          <ul className='flex flex-col gap-2 text-gray-600'>
            <li>Home</li>
            <li>About us</li>
            <li>Delivery</li>
            <li>Privacy policy</li>
          </ul>
        </div>

        <div>
          <p className='text-xl font-medium mb-5'>GET IN TOUCH</p>
          <ul className='flex flex-col gap-2 text-gray-600'>
            <li>+05 35 00 00 00</li>
            <li>medmeet@medmeet.com</li>
          </ul>
        </div>

      </div>

      <div>
        <hr />
        <p className='py-5 text-sm text-center'>Copyright © 2025 MedMeet Team @ AUI. All Rights Reserved.</p>
      </div>

    </div>
  )
}

export default Footer
