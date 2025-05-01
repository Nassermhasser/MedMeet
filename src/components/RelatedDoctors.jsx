import React, { useContext, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AppContext } from '../context/AppContext'

const RelatedDoctors = ({ speciality, docId }) => {
  const navigate = useNavigate()
  const { doctors } = useContext(AppContext)
  const [relDoc, setRelDoc] = useState([])

  useEffect(() => {
    if (doctors.length > 0 && speciality) {
      const doctorsData = doctors.filter((doc) => doc.speciality === speciality && doc._id !== docId)
      setRelDoc(doctorsData)
    }
  }, [doctors, speciality, docId])

  return (
    <div className='flex flex-col items-center gap-4 my-16 text-[#262626]'>
      <h1 className='text-3xl font-semibold'>Related Doctors</h1>
      <p className='sm:w-1/2 text-center text-sm text-gray-600'>
        Explore more professionals in the same field who are ready to assist you.
      </p>

      <div className='w-full grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 pt-6 px-4 sm:px-0'>
        {relDoc.map((item, index) => (
          <div
            key={index}
            onClick={() => { navigate(`/appointment/${item._id}`); scrollTo(0, 0) }}
            className='border border-[#C9D8FF] bg-white rounded-2xl overflow-hidden shadow-sm cursor-pointer hover:-translate-y-2 hover:shadow-md transition-all duration-300'
          >
            <img className='w-full h-48 object-cover bg-[#EAEFFF]' src={item.image} alt={item.name} />
            <div className='p-4 space-y-1'>
              <div className={`flex items-center gap-2 text-sm ${item.available ? 'text-green-500' : "text-gray-400"}`}>
                <span className={`w-2 h-2 rounded-full ${item.available ? 'bg-green-500' : "bg-gray-400"}`}></span>
                <p>{item.available ? 'Available' : 'Not Available'}</p>
              </div>
              <p className='text-lg font-medium text-[#262626]'>{item.name}</p>
              <p className='text-sm text-[#5C5C5C]'>{item.speciality}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default RelatedDoctors
