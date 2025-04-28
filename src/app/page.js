"use client";
import Head from "next/head";
import Image from "next/image";
import React, { useState } from "react";

export default function Home() {
  const [sidebarActive, setSidebarActive] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

  const toggleSidebar = () => setSidebarActive(!sidebarActive);
  const openModal = () => setModalOpen(true);
  const closeModal = () => setModalOpen(false);

  return (
    <>
      <Head>
        <title>Dashboard Example</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link
          href="https://fonts.googleapis.com/css?family=Beiruti"
          rel="stylesheet"
        />
      </Head>

      <div className="bg-gray-100 min-h-screen">
        {/* Navbar */}
        <header className="fixed top-0 left-0 w-full h-16 bg-[#15161B] text-white flex justify-between items-center p-3 z-50">
          <div className="flex items-center">
            <button onClick={toggleSidebar} className="mr-4 focus:outline-none">
              {/* ใช้ภาพ Hamburger icon */}
              <img src="/face.svg" alt="Menu Icon" className="w-13 h-13" />
            </button>
            <h1 className="text-5xl mt-1.5 font-normal break-words font-bebasneue">
              DASH BOARD Face Recognition
            </h1>
          </div>
          <div className="flex items-center">
            <span className="text-xl font-normal ml-4 break-words font-bebasneue">
              User Name
            </span>
            <span className="ml-2 inline-block w-11 h-11 rounded-full overflow-hidden">
              <img
                src="/1.jpg"
                alt="User Profile"
                className="w-full h-full object-cover"
              />
            </span>
          </div>
        </header>

        {/* Sidebar */}
        <aside
          className={`fixed top-16 left-0 h-[calc(100vh-4rem)] bg-[#15161B] z-40 transition-all duration-300 overflow-hidden ${
            sidebarActive ? "w-64" : "w-16"
          }`}
        >
          <div className="p-4 flex items-center">
            <button
              onClick={toggleSidebar}
              className="relative w-8 h-8 flex items-center justify-center focus:outline-none"
            >
              {/* แท่งที่ 1 */}
              <span
                className={`absolute h-0.5 w-6 bg-white rounded-full transform transition duration-300 ease-in-out ${
                  sidebarActive ? "rotate-45 translate-y-3.5" : "translate-y-1"
                }`}
              />
              {/* แท่งที่ 2 */}
              <span
                className={`absolute h-0.5 w-6 bg-white rounded-full transform transition duration-300 ease-in-out ${
                  sidebarActive
                    ? "opacity-0 -rotate-45 translate-y-3.5"
                    : "translate-y-3"
                }`}
              />
              {/* แท่งที่ 3 */}
              <span
                className={`absolute h-0.5 w-6 bg-white rounded-full transform transition duration-300 ease-in-out ${
                  sidebarActive ? "-rotate-45 translate-y-3.5" : "translate-y-5"
                }`}
              />
            </button>
          </div>

          <div className="border-t border-white opacity-50 mx-3"></div>

          <nav className="mt-4 flex flex-col space-y-4 px-3 items-start ml-2 font-[Beiruti]">
            <button className="flex items-center focus:outline-none">
              <img
                src="/cuida_outline.svg"
                alt="Dashboard Icon"
                className="w-6 h-10"
              />
              <span
                className={`ml-2 transition-opacity duration-300 text-white text-xl ${
                  sidebarActive ? "opacity-100" : "opacity-0"
                }`}
              >
                Dashboard
              </span>
            </button>
            <button className="flex items-center focus:outline-none">
              <img src="/users.svg" alt="Users Icon" className="w-6 h-10" />
              <span
                className={`ml-2 transition-opacity duration-300 text-white text-xl ${
                  sidebarActive ? "opacity-100" : "opacity-0"
                }`}
              >
                Users
              </span>
            </button>
            <button className="flex items-center focus:outline-none">
              <img
                src="/Information.svg"
                alt="Info Icon"
                className="w-6 h-10"
              />
              <span
                className={`ml-2 transition-opacity duration-300 text-white text-xl ${
                  sidebarActive ? "opacity-100" : "opacity-0"
                }`}
              >
                Information
              </span>
            </button>
            <button className="flex items-center focus:outline-none">
              <img
                src="/Setting.svg"
                alt="Settings Icon"
                className="w-6 h-10"
              />
              <span
                className={`ml-2 transition-opacity duration-300 text-white text-xl ${
                  sidebarActive ? "opacity-100" : "opacity-0"
                }`}
              >
                Settings
              </span>
            </button>
          </nav>

          <div className="absolute bottom-4 left-5">
            <button className="flex items-center focus:outline-none">
              <img
                src="/mingcute_exit-line.svg"
                alt="Logout Icon"
                className="w-6 h-10"
              />
              <span
                className={`ml-2 transition-opacity duration-300 text-white text-xl font-[Beiruti] ${
                  sidebarActive ? "opacity-100" : "opacity-0"
                }`}
              >
                Logout
              </span>
            </button>
          </div>
        </aside>

        {/* Main Content */}
        <main
          className={`pt-20 transition-all duration-300 ${
            sidebarActive ? "ml-64" : "ml-16"
          } p-4`}
        >
          <section>
            {/* Welcome Section */}
            <div className="mb-4">
              <h2 className="text-3xl font-bold font-[Beiruti] text-black">
                Welcome back, User Name!
              </h2>
              <p className="text-lg font-[Beiruti] text-gray-700">
                Here’s a brief summary of your attendance and face recognition
                status.
              </p>
            </div>

            {/* Stats */}
            <div className="flex gap-4 mb-6 font-[Beiruti]">
              <div className="flex-1 bg-green-200 rounded-lg p-4 shadow">
                <h3 className="text-2xl font-bold font-[Beiruti] text-black">
                  Present
                </h3>
                <p className="text-3xl font-[Beiruti] text-black">12</p>
              </div>
              <div className="flex-1 bg-yellow-200 rounded-lg p-4 shadow">
                <h3 className="text-2xl font-bold font-[Beiruti] text-black">
                  Late
                </h3>
                <p className="text-3xl font-[Beiruti] text-black">3</p>
              </div>
              <div className="flex-1 bg-pink-200 rounded-lg p-4 shadow">
                <h3 className="text-2xl font-bold font-[Beiruti] text-black">
                  Absent
                </h3>
                <p className="text-3xl font-[Beiruti] text-black">2</p>
              </div>
            </div>

            {/* Face Recognition Section */}
            <div className="w-full h-full flex gap-2 font-[Beiruti]">
              {/* Live Face Recognition */}
              <div className="flex-1 bg-white rounded-lg p-4 shadow">
                <h3 className="text-2xl font-bold mb-2 text-black">
                  Live Face Recognition
                </h3>
                <div className="h-[95%] bg-gray-200 flex items-center justify-center rounded-lg">
                  {/* ใส่ <img> ที่ src ชี้ไปยัง Flask server */}
                  <img
                    src="http://localhost:5000/video_feed"
                    alt="Live Face Recognition"
                    style={{ width: "100%", border: "1px solid #ccc" }}
                  />
                </div>
              </div>

              {/* Face Recognition History */}
              <div className="max-w-[20%] flex-1 bg-white rounded-lg p-4 shadow relative font-[Beiruti]">
                <div className="flex justify-between items-center mb-4 font-[Beiruti]">
                  <h3 className="text-2xl font-bold text-black font-[Beiruti]">
                    Face Recognition History
                  </h3>
                  <button className="text-xl focus:outline-none">⋮</button>
                </div>
                <div
                  id="historyList"
                  className="text-l overflow-y-auto max-h-[400px] pr-2"
                >
                  {/* History Items */}
                  <HistoryItem
                    name="Chayanon Chaikaew"
                    times="4 Times"
                    image="/1.jpg"
                  />
                  <HistoryItem
                    name="Jane Doe"
                    times="2 Times"
                    image="/1.jpg"
                  />
                  <HistoryItem
                    name="Jane Doe"
                    times="2 Times"
                    image="/1.jpg"
                  />
                  <HistoryItem
                    name="Jane Doe"
                    times="2 Times"
                    image="/1.jpg"
                  />
                  <HistoryItem
                    name="Jane Doe"
                    times="2 Times"
                    image="/1.jpg"
                  />
                  <HistoryItem
                    name="Jane Doe"
                    times="2 Times"
                    image="/1.jpg"
                  />
                </div>
                <button
                  onClick={openModal}
                  className="mt-4 border border-gray-300 rounded-lg px-4 py-2 text-blue-500 hover:bg-blue-100"
                >
                  Show More
                </button>

                {/* Modal */}
                {modalOpen && (
                  <div
                    onClick={closeModal}
                    className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50"
                  >
                    <div
                      onClick={(e) => e.stopPropagation()}
                      className="bg-white p-6 rounded-lg w-96 relative"
                    >
                      <button
                        onClick={closeModal}
                        className="absolute top-2 right-2 text-2xl"
                      >
                        &times;
                      </button>
                      <h2 className="text-2xl font-bold mb-4 text-black">
                        All Face Recognition History
                      </h2>
                      <div className="overflow-y-auto max-h-80">
                        {/* คัดลอก History Items */}
                        <HistoryItem
                          name="Chayanon Chaikaew"
                          times="4 Times"
                          image="/1.jpg"
                        />
                        <HistoryItem
                          name="Jane Doe"
                          times="2 Times"
                          image="/2.jpg"
                        />
                        <HistoryItem
                          name="Jane Doe"
                          times="2 Times"
                          image="/2.jpg"
                        />
                        <HistoryItem
                          name="Jane Doe"
                          times="2 Times"
                          image="/2.jpg"
                        />
                        <HistoryItem
                          name="Jane Doe"
                          times="2 Times"
                          image="/2.jpg"
                        />
                        <HistoryItem
                          name="Jane Doe"
                          times="2 Times"
                          image="/2.jpg"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>
        </main>
      </div>
    </>
  );
}

// ตัวอย่างฟังก์ชัน HistoryItem ที่ใช้แสดงประวัติใบหน้า
function HistoryItem({ name, times, image }) {
  return (
    <div className="flex items-center bg-white rounded-lg p-2 mb-2 border border-gray-200">
      <div className="w-10 h-10 rounded-full overflow-hidden mr-3">
        <img src={image} alt={name} className="w-full h-full object-cover" />
      </div>
      <div className="flex-1">
        <p className="font-semibold text-gray-800 text-lg">{name}</p>
        <p className="text-sm text-gray-500">{times}</p>
      </div>
      <button className="text-blue-500 text-base focus:outline-none">
        View
      </button>
    </div>
  );
}
