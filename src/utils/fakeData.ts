export const fakeCompanies = [
  {
    id: 1,
    taxNumber: '1234567890',
    taxOffice: 'İstanbul Vergi Dairesi',
    name: 'Arı İnşaat',
    description: 'İnşaat projeleri',
    manager: 'Ahmet Yılmaz',
    phone1: '555-123-4567',
    phone2: '555-987-6543',
    website: 'https://ariinsaat.com',
    email: 'info@ariinsaat.com',
    secondEmail: 'support@ariinsaat.com',
    city: 'İstanbul',
    district: 'Kadıköy',
  },
  {
    id: 2,
    taxNumber: '2345678901',
    taxOffice: 'Ankara Vergi Dairesi',
    name: 'Yeni İnşaat',
    description: 'Yapı ve mühendislik hizmetleri',
    manager: 'Fatma Demir',
    phone1: '555-234-5678',
    phone2: '555-876-5432',
    website: 'https://yeninsaat.com',
    email: 'info@yeninsaat.com',
    secondEmail: 'destek@yeninsaat.com',
    city: 'Ankara',
    district: 'Çankaya',
  },
  {
    id: 3,
    taxNumber: '3456789012',
    taxOffice: 'İzmir Vergi Dairesi',
    name: 'Kalem İnşaat',
    description: 'Altyapı çalışmaları',
    manager: 'Mehmet Çelik',
    phone1: '555-345-6789',
    phone2: '555-765-4321',
    website: 'https://kaleminsaat.com',
    email: 'info@kaleminsaat.com',
    secondEmail: 'support@kaleminsaat.com',
    city: 'İzmir',
    district: 'Konak',
  },
  {
    id: 4,
    taxNumber: '4567890123',
    taxOffice: 'Bursa Vergi Dairesi',
    name: 'Demir Yapı',
    description: 'Çelik yapı ve mühendislik',
    manager: 'Zeynep Aksoy',
    phone1: '555-456-7890',
    phone2: '555-654-3210',
    website: 'https://demiryapi.com',
    email: 'info@demiryapi.com',
    secondEmail: 'support@demiryapi.com',
    city: 'Bursa',
    district: 'Osmangazi',
  },
  {
    id: 5,
    taxNumber: '5678901234',
    taxOffice: 'Adana Vergi Dairesi',
    name: 'Beta Enerji',
    description: 'Enerji altyapı projeleri',
    manager: 'Ali Kılıç',
    phone1: '555-567-8901',
    phone2: '555-543-2109',
    website: 'https://betaenerji.com',
    email: 'info@betaenerji.com',
    secondEmail: 'destek@betaenerji.com',
    city: 'Adana',
    district: 'Seyhan',
  },
  {
    id: 6,
    taxNumber: '6789012345',
    taxOffice: 'Antalya Vergi Dairesi',
    name: 'Alpha Yazılım',
    description: 'Yazılım geliştirme ve teknoloji hizmetleri',
    manager: 'Cem Doğan',
    phone1: '555-678-9012',
    phone2: '555-432-1098',
    website: 'https://alphayazilim.com',
    email: 'info@alphayazilim.com',
    secondEmail: 'destek@alphayazilim.com',
    city: 'Antalya',
    district: 'Muratpaşa',
  },
  {
    id: 7,
    taxNumber: '7890123456',
    taxOffice: 'Konya Vergi Dairesi',
    name: 'Gamma Elektrik',
    description: 'Elektrik sistemleri ve çözümleri',
    manager: 'Mustafa Can',
    phone1: '555-789-0123',
    phone2: '555-321-0987',
    website: 'https://gammaelektrik.com',
    email: 'info@gammaelektrik.com',
    secondEmail: 'support@gammaelektrik.com',
    city: 'Konya',
    district: 'Selçuklu',
  },
  {
    id: 8,
    taxNumber: '8901234567',
    taxOffice: 'Mersin Vergi Dairesi',
    name: 'Delta Teknoloji',
    description: 'Yüksek teknoloji projeleri',
    manager: 'Ela Güneş',
    phone1: '555-890-1234',
    phone2: '555-210-9876',
    website: 'https://deltateknoloji.com',
    email: 'info@deltateknoloji.com',
    secondEmail: 'support@deltateknoloji.com',
    city: 'Mersin',
    district: 'Mezitli',
  },
  {
    id: 9,
    taxNumber: '9012345678',
    taxOffice: 'Gaziantep Vergi Dairesi',
    name: 'Sigma Lojistik',
    description: 'Lojistik ve taşıma hizmetleri',
    manager: 'Deniz Taş',
    phone1: '555-901-2345',
    phone2: '555-109-8765',
    website: 'https://sigmalojistik.com',
    email: 'info@sigmalojistik.com',
    secondEmail: 'destek@sigmalojistik.com',
    city: 'Gaziantep',
    district: 'Şahinbey',
  },
  {
    id: 10,
    taxNumber: '0123456789',
    taxOffice: 'Kocaeli Vergi Dairesi',
    name: 'Omega Finans',
    description: 'Finansal danışmanlık ve yönetim',
    manager: 'Hakan Demir',
    phone1: '555-012-3456',
    phone2: '555-098-7654',
    website: 'https://omegafinans.com',
    email: 'info@omegafinans.com',
    secondEmail: 'destek@omegafinans.com',
    city: 'Kocaeli',
    district: 'İzmit',
  },
];



export const fakeDepartments = [
  // Company 1
  { id: 1, name: 'Mühendislik', companyId: 1 },
  { id: 2, name: 'Proje Yönetimi', companyId: 1 },
  { id: 3, name: 'Muhasebe', companyId: 1 },
  { id: 4, name: 'Saha Operasyonları', companyId: 1 },
  { id: 5, name: 'IT Destek', companyId: 1 },
  { id: 6, name: 'Ar-Ge', companyId: 1 },
  { id: 7, name: 'Satış', companyId: 1 },
  { id: 8, name: 'Üretim', companyId: 1 },
  { id: 9, name: 'Finans', companyId: 1 },
  { id: 10, name: 'Teknik Destek', companyId: 1 },

  // Company 2
  { id: 11, name: 'Mühendislik', companyId: 2 },
  { id: 12, name: 'Proje Yönetimi', companyId: 2 },
  { id: 13, name: 'Muhasebe', companyId: 2 },
  { id: 14, name: 'Saha Operasyonları', companyId: 2 },
  { id: 15, name: 'IT Destek', companyId: 2 },
  { id: 16, name: 'Ar-Ge', companyId: 2 },
  { id: 17, name: 'Satış', companyId: 2 },
  { id: 18, name: 'Üretim', companyId: 2 },
  { id: 19, name: 'Finans', companyId: 2 },
  { id: 20, name: 'Teknik Destek', companyId: 2 },

  // Company 3
  { id: 21, name: 'Mühendislik', companyId: 3 },
  { id: 22, name: 'Proje Yönetimi', companyId: 3 },
  { id: 23, name: 'Muhasebe', companyId: 3 },
  { id: 24, name: 'Saha Operasyonları', companyId: 3 },
  { id: 25, name: 'IT Destek', companyId: 3 },
  { id: 26, name: 'Ar-Ge', companyId: 3 },
  { id: 27, name: 'Satış', companyId: 3 },
  { id: 28, name: 'Üretim', companyId: 3 },
  { id: 29, name: 'Finans', companyId: 3 },
  { id: 30, name: 'Teknik Destek', companyId: 3 },

  // Company 4
  { id: 31, name: 'Mühendislik', companyId: 4 },
  { id: 32, name: 'Proje Yönetimi', companyId: 4 },
  { id: 33, name: 'Muhasebe', companyId: 4 },
  { id: 34, name: 'Saha Operasyonları', companyId: 4 },
  { id: 35, name: 'IT Destek', companyId: 4 },
  { id: 36, name: 'Ar-Ge', companyId: 4 },
  { id: 37, name: 'Satış', companyId: 4 },
  { id: 38, name: 'Üretim', companyId: 4 },
  { id: 39, name: 'Finans', companyId: 4 },
  { id: 40, name: 'Teknik Destek', companyId: 4 },

  // Company 5
  { id: 41, name: 'Mühendislik', companyId: 5 },
  { id: 42, name: 'Proje Yönetimi', companyId: 5 },
  { id: 43, name: 'Muhasebe', companyId: 5 },
  { id: 44, name: 'Saha Operasyonları', companyId: 5 },
  { id: 45, name: 'IT Destek', companyId: 5 },
  { id: 46, name: 'Ar-Ge', companyId: 5 },
  { id: 47, name: 'Satış', companyId: 5 },
  { id: 48, name: 'Üretim', companyId: 5 },
  { id: 49, name: 'Finans', companyId: 5 },
  { id: 50, name: 'Teknik Destek', companyId: 5 },

  // Company 6
  { id: 51, name: 'Mühendislik', companyId: 6 },
  { id: 52, name: 'Proje Yönetimi', companyId: 6 },
  { id: 53, name: 'Muhasebe', companyId: 6 },
  { id: 54, name: 'Saha Operasyonları', companyId: 6 },
  { id: 55, name: 'IT Destek', companyId: 6 },
  { id: 56, name: 'Ar-Ge', companyId: 6 },
  { id: 57, name: 'Satış', companyId: 6 },
  { id: 58, name: 'Üretim', companyId: 6 },
  { id: 59, name: 'Finans', companyId: 6 },
  { id: 60, name: 'Teknik Destek', companyId: 6 },

  // Company 7
  { id: 61, name: 'Mühendislik', companyId: 7 },
  { id: 62, name: 'Proje Yönetimi', companyId: 7 },
  { id: 63, name: 'Muhasebe', companyId: 7 },
  { id: 64, name: 'Saha Operasyonları', companyId: 7 },
  { id: 65, name: 'IT Destek', companyId: 7 },
  { id: 66, name: 'Ar-Ge', companyId: 7 },
  { id: 67, name: 'Satış', companyId: 7 },
  { id: 68, name: 'Üretim', companyId: 7 },
  { id: 69, name: 'Finans', companyId: 7 },
  { id: 70, name: 'Teknik Destek', companyId: 7 },

  // Company 8
  { id: 71, name: 'Mühendislik', companyId: 8 },
  { id: 72, name: 'Proje Yönetimi', companyId: 8 },
  { id: 73, name: 'Muhasebe', companyId: 8 },
  { id: 74, name: 'Saha Operasyonları', companyId: 8 },
  { id: 75, name: 'IT Destek', companyId: 8 },
  { id: 76, name: 'Ar-Ge', companyId: 8 },
  { id: 77, name: 'Satış', companyId: 8 },
  { id: 78, name: 'Üretim', companyId: 8 },
  { id: 79, name: 'Finans', companyId: 8 },
  { id: 80, name: 'Teknik Destek', companyId: 8 },

  // Company 9
  { id: 81, name: 'Mühendislik', companyId: 9 },
  { id: 82, name: 'Proje Yönetimi', companyId: 9 },
  { id: 83, name: 'Muhasebe', companyId: 9 },
  { id: 84, name: 'Saha Operasyonları', companyId: 9 },
  { id: 85, name: 'IT Destek', companyId: 9 },
  { id: 86, name: 'Ar-Ge', companyId: 9 },
  { id: 87, name: 'Satış', companyId: 9 },
  { id: 88, name: 'Üretim', companyId: 9 },
  { id: 89, name: 'Finans', companyId: 9 },
  { id: 90, name: 'Teknik Destek', companyId: 9 },

  // Company 10
  { id: 91, name: 'Mühendislik', companyId: 10 },
  { id: 92, name: 'Proje Yönetimi', companyId: 10 },
  { id: 93, name: 'Muhasebe', companyId: 10 },
  { id: 94, name: 'Saha Operasyonları', companyId: 10 },
  { id: 95, name: 'IT Destek', companyId: 10 },
  { id: 96, name: 'Ar-Ge', companyId: 10 },
  { id: 97, name: 'Satış', companyId: 10 },
  { id: 98, name: 'Üretim', companyId: 10 },
  { id: 99, name: 'Finans', companyId: 10 },
  { id: 100, name: 'Teknik Destek', companyId: 10 },
];

export const fakeUsers = [
  // Departman 1: Mühendislik (10 kullanıcı)
  { id: 1, name: 'Ahmet Yılmaz', role: 'Baş Mühendis', email: 'ahmet.yilmaz@ariinsaat.com', phone: '555-123-4567', departmentId: 1, companyId: 1 },
  { id: 2, name: 'Ayşe Kaya', role: 'Mimarlık Uzmanı', email: 'ayse.kaya@ariinsaat.com', phone: '555-234-5678', departmentId: 1, companyId: 1 },
  { id: 3, name: 'Mehmet Öz', role: 'Yapı Mühendisi', email: 'mehmet.oz@ariinsaat.com', phone: '555-345-6789', departmentId: 1, companyId: 1 },
  { id: 4, name: 'Elif Demir', role: 'Elektrik Mühendisi', email: 'elif.demir@ariinsaat.com', phone: '555-456-7890', departmentId: 1, companyId: 1 },
  { id: 5, name: 'Mustafa Can', role: 'Makine Mühendisi', email: 'mustafa.can@ariinsaat.com', phone: '555-567-8901', departmentId: 1, companyId: 1 },
  { id: 6, name: 'Zeynep Aksoy', role: 'Proje Mühendisi', email: 'zeynep.aksoy@ariinsaat.com', phone: '555-678-9012', departmentId: 1, companyId: 1 },
  { id: 7, name: 'Ali Veli', role: 'İnşaat Mühendisi', email: 'ali.veli@ariinsaat.com', phone: '555-789-0123', departmentId: 1, companyId: 1 },
  { id: 8, name: 'Fatma Aydın', role: 'Çevre Mühendisi', email: 'fatma.aydin@ariinsaat.com', phone: '555-890-1234', departmentId: 1, companyId: 1 },
  { id: 9, name: 'Hasan Şahin', role: 'Jeoloji Mühendisi', email: 'hasan.sahin@ariinsaat.com', phone: '555-901-2345', departmentId: 1, companyId: 1 },
  { id: 10, name: 'Selin Yıldız', role: 'İnşaat Teknisyeni', email: 'selin.yildiz@ariinsaat.com', phone: '555-012-3456', departmentId: 1, companyId: 1 },

  // Departman 2: Proje Yönetimi (10 kullanıcı)
  { id: 11, name: 'Burak Sarı', role: 'Proje Müdürü', email: 'burak.sari@ariinsaat.com', phone: '555-123-4568', departmentId: 2, companyId: 1 },
  { id: 12, name: 'Derya Toprak', role: 'Proje Koordinatörü', email: 'derya.toprak@ariinsaat.com', phone: '555-234-5679', departmentId: 2, companyId: 1 },
  { id: 13, name: 'Emre Korkmaz', role: 'Proje Asistanı', email: 'emre.korkmaz@ariinsaat.com', phone: '555-345-6790', departmentId: 2, companyId: 1 },
  { id: 14, name: 'Gizem Yıldırım', role: 'Risk Yönetimi Uzmanı', email: 'gizem.yildirim@ariinsaat.com', phone: '555-456-7891', departmentId: 2, companyId: 1 },
  { id: 15, name: 'Hakan Doğan', role: 'Kalite Güvence Yöneticisi', email: 'hakan.dogan@ariinsaat.com', phone: '555-567-8902', departmentId: 2, companyId: 1 },
  { id: 16, name: 'İlayda Acar', role: 'Zaman Yönetimi Uzmanı', email: 'ilayda.acar@ariinsaat.com', phone: '555-678-9013', departmentId: 2, companyId: 1 },
  { id: 17, name: 'Kaan Türkmen', role: 'Bütçe Analisti', email: 'kaan.turkmen@ariinsaat.com', phone: '555-789-0124', departmentId: 2, companyId: 1 },
  { id: 18, name: 'Lale Gül', role: 'İletişim Sorumlusu', email: 'lale.gul@ariinsaat.com', phone: '555-890-1235', departmentId: 2, companyId: 1 },
  { id: 19, name: 'Murat Kaplan', role: 'Planlama Uzmanı', email: 'murat.kaplan@ariinsaat.com', phone: '555-901-2346', departmentId: 2, companyId: 1 },
  { id: 20, name: 'Nesrin Efe', role: 'Dokümantasyon Uzmanı', email: 'nesrin.efe@ariinsaat.com', phone: '555-012-3457', departmentId: 2, companyId: 1 },

  // Departman 3: Muhasebe (10 kullanıcı)
  { id: 21, name: 'Fatma Demir', role: 'Muhasebeci', email: 'fatma.demir@yeninsaat.com', phone: '555-456-7890', departmentId: 3, companyId: 2 },
  { id: 22, name: 'Ahmet Can', role: 'Finans Muhasebecisi', email: 'ahmet.can@yeninsaat.com', phone: '555-567-8903', departmentId: 3, companyId: 2 },
  { id: 23, name: 'Buse Kaya', role: 'Vergi Uzmanı', email: 'buse.kaya@yeninsaat.com', phone: '555-678-9014', departmentId: 3, companyId: 2 },
  { id: 24, name: 'Cem Öz', role: 'Bütçe Kontrolörü', email: 'cem.oz@yeninsaat.com', phone: '555-789-0125', departmentId: 3, companyId: 2 },
  { id: 25, name: 'Deniz Arslan', role: 'Muhasebe Asistanı', email: 'deniz.arslan@yeninsaat.com', phone: '555-890-1236', departmentId: 3, companyId: 2 },
  { id: 26, name: 'Elif Yıldız', role: 'Mali Analist', email: 'elif.yildiz@yeninsaat.com', phone: '555-901-2347', departmentId: 3, companyId: 2 },
  { id: 27, name: 'Furkan Demirtaş', role: 'Hesap Uzmanı', email: 'furkan.demirtas@yeninsaat.com', phone: '555-012-3458', departmentId: 3, companyId: 2 },
  { id: 28, name: 'Gülşah Çelik', role: 'Muhasebe Müdürü', email: 'gulsen.celik@yeninsaat.com', phone: '555-123-4569', departmentId: 3, companyId: 2 },
  { id: 29, name: 'Hüseyin Aydın', role: 'Entegre Muhasebe Uzmanı', email: 'huseyin.aydin@yeninsaat.com', phone: '555-234-5670', departmentId: 3, companyId: 2 },
  { id: 30, name: 'İrem Toprak', role: 'Mali Raporlama Uzmanı', email: 'irem.toprak@yeninsaat.com', phone: '555-345-6780', departmentId: 3, companyId: 2 },

  // Departman 4: Saha Operasyonları (10 kullanıcı)
  { id: 31, name: 'Deniz Taş', role: 'Saha Operatörü', email: 'deniz.tas@kaleminsaat.com', phone: '555-678-9012', departmentId: 4, companyId: 3 },
  { id: 32, name: 'Erdem Kılıç', role: 'Saha Müdürü', email: 'erdem.kilic@kaleminsaat.com', phone: '555-789-0126', departmentId: 4, companyId: 3 },
  { id: 33, name: 'Fatih Yılmaz', role: 'Operasyon Sorumlusu', email: 'fatih.yilmaz@kaleminsaat.com', phone: '555-890-1237', departmentId: 4, companyId: 3 },
  { id: 34, name: 'Gamze Acar', role: 'Saha Teknisyeni', email: 'gamze.acar@kaleminsaat.com', phone: '555-901-2348', departmentId: 4, companyId: 3 },
  { id: 35, name: 'Hakan Şimşek', role: 'İnşaat Görevlisi', email: 'hakan.simsek@kaleminsaat.com', phone: '555-012-3459', departmentId: 4, companyId: 3 },
  { id: 36, name: 'İbrahim Efe', role: 'Saha Asistanı', email: 'ibrahim.efe@kaleminsaat.com', phone: '555-123-4570', departmentId: 4, companyId: 3 },
  { id: 37, name: 'Jale Güneş', role: 'Saha Denetçisi', email: 'jale.gunes@kaleminsaat.com', phone: '555-234-5671', departmentId: 4, companyId: 3 },
  { id: 38, name: 'Kemal Arslan', role: 'Malzeme Sorumlusu', email: 'kemal.arslan@kaleminsaat.com', phone: '555-345-6781', departmentId: 4, companyId: 3 },
  { id: 39, name: 'Leman Çelik', role: 'Saha Planlamacısı', email: 'leman.celik@kaleminsaat.com', phone: '555-456-7892', departmentId: 4, companyId: 3 },
  { id: 40, name: 'Murat Yıldırım', role: 'İş Güvenliği Uzmanı', email: 'murat.yildirim@kaleminsaat.com', phone: '555-567-8904', departmentId: 4, companyId: 3 },

  // Departman 5: IT Destek (10 kullanıcı)
  { id: 41, name: 'Ece Kara', role: 'IT Uzmanı', email: 'ece.kara@alphayazilim.com', phone: '555-789-0123', departmentId: 5, companyId: 6 },
  { id: 42, name: 'Barış Demir', role: 'Ağ Yöneticisi', email: 'baris.demir@alphayazilim.com', phone: '555-890-1238', departmentId: 5, companyId: 6 },
  { id: 43, name: 'Cansu Aydın', role: 'Sistem Analisti', email: 'cansu.aydin@alphayazilim.com', phone: '555-901-2349', departmentId: 5, companyId: 6 },
  { id: 44, name: 'Deniz Yılmaz', role: 'Veritabanı Yöneticisi', email: 'deniz.yilmaz@alphayazilim.com', phone: '555-012-3460', departmentId: 5, companyId: 6 },
  { id: 45, name: 'Emre Şahin', role: 'Yazılım Destek Uzmanı', email: 'emre.sahin@alphayazilim.com', phone: '555-123-4571', departmentId: 5, companyId: 6 },
  { id: 46, name: 'Funda Kılıç', role: 'Teknik Destek Mühendisi', email: 'funda.kilic@alphayazilim.com', phone: '555-234-5672', departmentId: 5, companyId: 6 },
  { id: 47, name: 'Gökhan Arslan', role: 'IT Güvenlik Uzmanı', email: 'gokhan.arslan@alphayazilim.com', phone: '555-345-6782', departmentId: 5, companyId: 6 },
  { id: 48, name: 'Hande Efe', role: 'Uygulama Yöneticisi', email: 'hande.efe@alphayazilim.com', phone: '555-456-7893', departmentId: 5, companyId: 6 },
  { id: 49, name: 'İlker Toprak', role: 'Bulut Bilişim Uzmanı', email: 'ilker.toprak@alphayazilim.com', phone: '555-567-8905', departmentId: 5, companyId: 6 },
  { id: 50, name: 'Jülide Güneş', role: 'IT Proje Yöneticisi', email: 'julide.gunes@alphayazilim.com', phone: '555-678-9015', departmentId: 5, companyId: 6 },

  // Departman 6: Ar-Ge (10 kullanıcı)
  { id: 51, name: 'Ali Kılıç', role: 'Enerji Mühendisi', email: 'ali.kilic@betaenerji.com', phone: '555-890-1234', departmentId: 7, companyId: 5 },
  { id: 52, name: 'Büşra Yılmaz', role: 'Ar-Ge Uzmanı', email: 'buse.yilmaz@demiryapi.com', phone: '555-901-2350', departmentId: 6, companyId: 4 },
  { id: 53, name: 'Canan Acar', role: 'Ürün Geliştirme Mühendisi', email: 'canan.acar@demiryapi.com', phone: '555-012-3461', departmentId: 6, companyId: 4 },
  { id: 54, name: 'Davut Şimşek', role: 'Ar-Ge Analisti', email: 'davut.simsek@demiryapi.com', phone: '555-123-4572', departmentId: 6, companyId: 4 },
  { id: 55, name: 'Elif Demirtaş', role: 'Prototip Geliştirici', email: 'elif.demirtas@demiryapi.com', phone: '555-234-5673', departmentId: 6, companyId: 4 },
  { id: 56, name: 'Furkan Yıldırım', role: 'Malzeme Bilimcisi', email: 'furkan.yildirim@demiryapi.com', phone: '555-345-6783', departmentId: 6, companyId: 4 },
  { id: 57, name: 'Gizem Korkmaz', role: 'İnovasyon Uzmanı', email: 'gizem.korkmaz@demiryapi.com', phone: '555-456-7894', departmentId: 6, companyId: 4 },
  { id: 58, name: 'Hakan Aydın', role: 'Araştırma Bilimcisi', email: 'hakan.aydin@demiryapi.com', phone: '555-567-8906', departmentId: 6, companyId: 4 },
  { id: 59, name: 'İrem Arslan', role: 'Veri Analisti', email: 'irem.arslan@demiryapi.com', phone: '555-678-9016', departmentId: 6, companyId: 4 },
  { id: 60, name: 'Kemal Güneş', role: 'Proje Asistanı', email: 'kemal.gunes@demiryapi.com', phone: '555-789-0127', departmentId: 6, companyId: 4 },

  // Departman 7: Satış (10 kullanıcı)
  { id: 61, name: 'Lale Yılmaz', role: 'Satış Müdürü', email: 'lale.yilmaz@betaenerji.com', phone: '555-890-1235', departmentId: 7, companyId: 5 },
  { id: 62, name: 'Murat Çelik', role: 'Satış Temsilcisi', email: 'murat.celik@betaenerji.com', phone: '555-901-2351', departmentId: 7, companyId: 5 },
  { id: 63, name: 'Nazan Acar', role: 'Büyük Müşteri Satış Uzmanı', email: 'nazan.acar@betaenerji.com', phone: '555-012-3462', departmentId: 7, companyId: 5 },
  { id: 64, name: 'Okan Şimşek', role: 'Satış Destek Uzmanı', email: 'okan.simsek@betaenerji.com', phone: '555-123-4573', departmentId: 7, companyId: 5 },
  { id: 65, name: 'Pelin Yıldırım', role: 'Satış Analisti', email: 'pelin.yildirim@betaenerji.com', phone: '555-234-5674', departmentId: 7, companyId: 5 },
  { id: 66, name: 'Recep Korkmaz', role: 'Pazarlama ve Satış Uzmanı', email: 'recep.korkmaz@betaenerji.com', phone: '555-345-6784', departmentId: 7, companyId: 5 },
  { id: 67, name: 'Seda Aydın', role: 'Müşteri İlişkileri Yöneticisi', email: 'seda.aydin@betaenerji.com', phone: '555-456-7895', departmentId: 7, companyId: 5 },
  { id: 68, name: 'Taha Arslan', role: 'Satış Stratejisti', email: 'taha.arslan@betaenerji.com', phone: '555-567-8907', departmentId: 7, companyId: 5 },
  { id: 69, name: 'Uğur Güneş', role: 'Satış Operasyon Uzmanı', email: 'ugur.gunes@betaenerji.com', phone: '555-678-9017', departmentId: 7, companyId: 5 },
  { id: 70, name: 'Veli Demirtaş', role: 'Satış Eğitmeni', email: 'veli.demirtas@betaenerji.com', phone: '555-789-0128', departmentId: 7, companyId: 5 },

  // Departman 8: Üretim (10 kullanıcı)
  { id: 71, name: 'Mustafa Can', role: 'Elektrik Mühendisi', email: 'mustafa.can@gammaelektrik.com', phone: '555-111-2222', departmentId: 8, companyId: 7 },
  { id: 72, name: 'Ayşe Demir', role: 'Üretim Müdürü', email: 'ayse.demir@gammaelektrik.com', phone: '555-222-3333', departmentId: 8, companyId: 7 },
  { id: 73, name: 'Burak Yılmaz', role: 'Üretim Planlayıcısı', email: 'burak.yilmaz@gammaelektrik.com', phone: '555-333-4444', departmentId: 8, companyId: 7 },
  { id: 74, name: 'Cansu Aydın', role: 'Kalite Kontrol Uzmanı', email: 'cansu.aydin@gammaelektrik.com', phone: '555-444-5555', departmentId: 8, companyId: 7 },
  { id: 75, name: 'Deniz Kılıç', role: 'Makine Operatörü', email: 'deniz.kilic@gammaelektrik.com', phone: '555-555-6666', departmentId: 8, companyId: 7 },
  { id: 76, name: 'Eren Şahin', role: 'Üretim Asistanı', email: 'eren.sahin@gammaelektrik.com', phone: '555-666-7777', departmentId: 8, companyId: 7 },
  { id: 77, name: 'Furkan Yıldırım', role: 'Üretim Teknikeri', email: 'furkan.yildirim@gammaelektrik.com', phone: '555-777-8888', departmentId: 8, companyId: 7 },
  { id: 78, name: 'Gizem Arslan', role: 'Üretim Analisti', email: 'gizem.arslan@gammaelektrik.com', phone: '555-888-9999', departmentId: 8, companyId: 7 },
  { id: 79, name: 'Hakan Aydın', role: 'Üretim Denetçisi', email: 'hakan.aydin@gammaelektrik.com', phone: '555-999-0000', departmentId: 8, companyId: 7 },
  { id: 80, name: 'İrem Toprak', role: 'Üretim Koordinatörü', email: 'irem.toprak@gammaelektrik.com', phone: '555-000-1111', departmentId: 8, companyId: 7 },

  // Departman 9: Finans (10 kullanıcı)
  { id: 81, name: 'Hakan Demir', role: 'Finansal Danışman', email: 'hakan.demir@omegafinans.com', phone: '555-112-3344', departmentId: 9, companyId: 10 },
  { id: 82, name: 'İlayda Acar', role: 'Finans Analisti', email: 'ilayda.acar@omegafinans.com', phone: '555-223-4455', departmentId: 9, companyId: 10 },
  { id: 83, name: 'Kaan Türkmen', role: 'Bütçe Uzmanı', email: 'kaan.turkmen@omegafinans.com', phone: '555-334-5566', departmentId: 9, companyId: 10 },
  { id: 84, name: 'Lale Gül', role: 'Yatırım Danışmanı', email: 'lale.gul@omegafinans.com', phone: '555-445-6677', departmentId: 9, companyId: 10 },
  { id: 85, name: 'Murat Kaplan', role: 'Muhasebe Müdürü', email: 'murat.kaplan@omegafinans.com', phone: '555-556-7788', departmentId: 9, companyId: 10 },
  { id: 86, name: 'Nesrin Efe', role: 'Finans Müdürü', email: 'nesrin.efe@omegafinans.com', phone: '555-667-8899', departmentId: 9, companyId: 10 },
  { id: 87, name: 'Ozan Yılmaz', role: 'Kredi Analisti', email: 'ozan.yilmaz@omegafinans.com', phone: '555-778-9900', departmentId: 9, companyId: 10 },
  { id: 88, name: 'Pelin Yıldırım', role: 'Risk Yönetimi Uzmanı', email: 'pelin.yildirim@omegafinans.com', phone: '555-889-0011', departmentId: 9, companyId: 10 },
  { id: 89, name: 'Recep Korkmaz', role: 'Finansal Raporlama Uzmanı', email: 'recep.korkmaz@omegafinans.com', phone: '555-990-1122', departmentId: 9, companyId: 10 },
  { id: 90, name: 'Seda Aydın', role: 'Finans Operasyon Uzmanı', email: 'seda.aydin@omegafinans.com', phone: '555-101-2132', departmentId: 9, companyId: 10 },

  // Departman 10: Teknik Destek (10 kullanıcı)
  { id: 91, name: 'Mustafa Can', role: 'Elektrik Mühendisi', email: 'mustafa.can@gammadestek.com', phone: '555-111-2223', departmentId: 10, companyId: 8 },
  { id: 92, name: 'Ayşe Demir', role: 'Teknik Destek Uzmanı', email: 'ayse.demir@gammadestek.com', phone: '555-222-3334', departmentId: 10, companyId: 8 },
  { id: 93, name: 'Burak Yılmaz', role: 'Destek Mühendisi', email: 'burak.yilmaz@gammadestek.com', phone: '555-333-4445', departmentId: 10, companyId: 8 },
  { id: 94, name: 'Cansu Aydın', role: 'Teknik Destek Asistanı', email: 'cansu.aydin@gammadestek.com', phone: '555-444-5556', departmentId: 10, companyId: 8 },
  { id: 95, name: 'Deniz Kılıç', role: 'IT Destek Uzmanı', email: 'deniz.kilic@gammadestek.com', phone: '555-555-6667', departmentId: 10, companyId: 8 },
  { id: 96, name: 'Eren Şahin', role: 'Teknik Analist', email: 'eren.sahin@gammadestek.com', phone: '555-666-7778', departmentId: 10, companyId: 8 },
  { id: 97, name: 'Furkan Yıldırım', role: 'Destek Operasyon Uzmanı', email: 'furkan.yildirim@gammadestek.com', phone: '555-777-8889', departmentId: 10, companyId: 8 },
  { id: 98, name: 'Gizem Arslan', role: 'Teknik Koordinatör', email: 'gizem.arslan@gammadestek.com', phone: '555-888-9990', departmentId: 10, companyId: 8 },
  { id: 99, name: 'Hakan Aydın', role: 'Destek Planlayıcısı', email: 'hakan.aydin@gammadestek.com', phone: '555-999-0001', departmentId: 10, companyId: 8 },
  { id: 100, name: 'İrem Toprak', role: 'Teknik Destek Müdürü', email: 'irem.toprak@gammadestek.com', phone: '555-000-1112', departmentId: 10, companyId: 8 },

  // Departman 11: Lojistik Yönetimi (10 kullanıcı)
  { id: 101, name: 'Jale Güneş', role: 'Lojistik Müdürü', email: 'jale.gunes@sigmalojistik.com', phone: '555-112-3345', departmentId: 11, companyId: 9 },
  { id: 102, name: 'Kemal Arslan', role: 'Taşıma Koordinatörü', email: 'kemal.arslan@sigmalojistik.com', phone: '555-223-4456', departmentId: 11, companyId: 9 },
  { id: 103, name: 'Leman Çelik', role: 'Depo Yöneticisi', email: 'leman.celik@sigmalojistik.com', phone: '555-334-5567', departmentId: 11, companyId: 9 },
  { id: 104, name: 'Murat Yıldırım', role: 'Nakliye Uzmanı', email: 'murat.yildirim@sigmalojistik.com', phone: '555-445-6678', departmentId: 11, companyId: 9 },
  { id: 105, name: 'Nesrin Efe', role: 'Lojistik Analisti', email: 'nesrin.efe@sigmalojistik.com', phone: '555-556-7789', departmentId: 11, companyId: 9 },
  { id: 106, name: 'Ozan Yılmaz', role: 'Tedarik Zinciri Uzmanı', email: 'ozan.yilmaz@sigmalojistik.com', phone: '555-667-8890', departmentId: 11, companyId: 9 },
  { id: 107, name: 'Pelin Yıldırım', role: 'Lojistik Planlayıcısı', email: 'pelin.yildirim@sigmalojistik.com', phone: '555-778-9901', departmentId: 11, companyId: 9 },
  { id: 108, name: 'Recep Korkmaz', role: 'Lojistik Operasyon Uzmanı', email: 'recep.korkmaz@sigmalojistik.com', phone: '555-889-0012', departmentId: 11, companyId: 9 },
  { id: 109, name: 'Seda Aydın', role: 'Envanter Yönetimi Uzmanı', email: 'seda.aydin@sigmalojistik.com', phone: '555-990-1123', departmentId: 11, companyId: 9 },
  { id: 110, name: 'Taha Arslan', role: 'Lojistik Destek Uzmanı', email: 'taha.arslan@sigmalojistik.com', phone: '555-101-2133', departmentId: 11, companyId: 9 },

  // Departman 12: Müşteri Hizmetleri (10 kullanıcı)
  { id: 111, name: 'Uğur Güneş', role: 'Müşteri Hizmetleri Müdürü', email: 'ugur.gunes@deltateknoloji.com', phone: '555-112-3346', departmentId: 12, companyId: 8 },
  { id: 112, name: 'Veli Demirtaş', role: 'Müşteri Temsilcisi', email: 'veli.demirtas@deltateknoloji.com', phone: '555-223-4457', departmentId: 12, companyId: 8 },
  { id: 113, name: 'Yasemin Yıldız', role: 'Destek Uzmanı', email: 'yasemin.yildiz@deltateknoloji.com', phone: '555-334-5568', departmentId: 12, companyId: 8 },
  { id: 114, name: 'Zeki Acar', role: 'Çağrı Merkezi Sorumlusu', email: 'zeki.acar@deltateknoloji.com', phone: '555-445-6679', departmentId: 12, companyId: 8 },
  { id: 115, name: 'Ahmet Efe', role: 'Müşteri İlişkileri Uzmanı', email: 'ahmet.efe@deltateknoloji.com', phone: '555-556-7790', departmentId: 12, companyId: 8 },
  { id: 116, name: 'Büşra Kılıç', role: 'Müşteri Destek Uzmanı', email: 'buse.kilic@deltateknoloji.com', phone: '555-667-8891', departmentId: 12, companyId: 8 },
  { id: 117, name: 'Cem Şahin', role: 'Müşteri Danışmanı', email: 'cem.sahin@deltateknoloji.com', phone: '555-778-9902', departmentId: 12, companyId: 8 },
  { id: 118, name: 'Deniz Yıldırım', role: 'Müşteri Hizmetleri Asistanı', email: 'deniz.yildirim@deltateknoloji.com', phone: '555-889-0013', departmentId: 12, companyId: 8 },
  { id: 119, name: 'Elif Arslan', role: 'Müşteri Memnuniyeti Uzmanı', email: 'elif.arslan@deltateknoloji.com', phone: '555-990-1124', departmentId: 12, companyId: 8 },
  { id: 120, name: 'Furkan Aydın', role: 'Müşteri Hizmetleri Analisti', email: 'furkan.aydin@deltateknoloji.com', phone: '555-101-2134', departmentId: 12, companyId: 8 },

  // Departman 13: Halkla İlişkiler (10 kullanıcı)
  { id: 121, name: 'Gamze Kılıç', role: 'Halkla İlişkiler Müdürü', email: 'gamze.kilic@demiryapi.com', phone: '555-112-3347', departmentId: 13, companyId: 4 },
  { id: 122, name: 'Hakan Şimşek', role: 'PR Uzmanı', email: 'hakan.simsek@demiryapi.com', phone: '555-223-4458', departmentId: 13, companyId: 4 },
  { id: 123, name: 'İlayda Efe', role: 'Medya İlişkileri Sorumlusu', email: 'ilayda.efe@demiryapi.com', phone: '555-334-5569', departmentId: 13, companyId: 4 },
  { id: 124, name: 'Kaan Acar', role: 'İç İletişim Uzmanı', email: 'kaan.acar@demiryapi.com', phone: '555-445-6680', departmentId: 13, companyId: 4 },
  { id: 125, name: 'Lale Yıldız', role: 'Dış İletişim Uzmanı', email: 'lale.yildiz@demiryapi.com', phone: '555-556-7791', departmentId: 13, companyId: 4 },
  { id: 126, name: 'Murat Kaplan', role: 'Etkinlik Koordinatörü', email: 'murat.kaplan@demiryapi.com', phone: '555-667-8892', departmentId: 13, companyId: 4 },
  { id: 127, name: 'Nesrin Demir', role: 'Sosyal Medya Uzmanı', email: 'nesrin.demir@demiryapi.com', phone: '555-778-9903', departmentId: 13, companyId: 4 },
  { id: 128, name: 'Ozan Arslan', role: 'Kurumsal İletişim Uzmanı', email: 'ozan.arslan@demiryapi.com', phone: '555-889-0014', departmentId: 13, companyId: 4 },
  { id: 129, name: 'Pelin Yıldırım', role: 'Medya İlişkileri Analisti', email: 'pelin.yildirim@demiryapi.com', phone: '555-990-1125', departmentId: 13, companyId: 4 },
  { id: 130, name: 'Recep Korkmaz', role: 'Halkla İlişkiler Asistanı', email: 'recep.korkmaz@demiryapi.com', phone: '555-101-2135', departmentId: 13, companyId: 4 },

  // Departman 14: Kalite Kontrol (10 kullanıcı)
  { id: 131, name: 'Seda Aydın', role: 'Kalite Kontrol Müdürü', email: 'seda.aydin@yeninsaat.com', phone: '555-112-3348', departmentId: 14, companyId: 3 },
  { id: 132, name: 'Taha Arslan', role: 'Kalite Kontrol Uzmanı', email: 'taha.arslan@yeninsaat.com', phone: '555-223-4459', departmentId: 14, companyId: 3 },
  { id: 133, name: 'Uğur Güneş', role: 'Denetim Uzmanı', email: 'ugur.gunes@yeninsaat.com', phone: '555-334-5570', departmentId: 14, companyId: 3 },
  { id: 134, name: 'Veli Demirtaş', role: 'Kalite Güvence Uzmanı', email: 'veli.demirtas@yeninsaat.com', phone: '555-445-6681', departmentId: 14, companyId: 3 },
  { id: 135, name: 'Yasemin Yıldız', role: 'İç Denetim Uzmanı', email: 'yasemin.yildiz@yeninsaat.com', phone: '555-556-7792', departmentId: 14, companyId: 3 },
  { id: 136, name: 'Zeki Acar', role: 'Kalite Analisti', email: 'zeki.acar@yeninsaat.com', phone: '555-667-8893', departmentId: 14, companyId: 3 },
  { id: 137, name: 'Ahmet Efe', role: 'Proses Kalite Uzmanı', email: 'ahmet.efe@yeninsaat.com', phone: '555-778-9904', departmentId: 14, companyId: 3 },
  { id: 138, name: 'Büşra Kılıç', role: 'Kalite Denetçisi', email: 'buse.kilic@yeninsaat.com', phone: '555-889-0015', departmentId: 14, companyId: 3 },
  { id: 139, name: 'Cem Şahin', role: 'Kalite İyileştirme Uzmanı', email: 'cem.sahin@yeninsaat.com', phone: '555-990-1126', departmentId: 14, companyId: 3 },
  { id: 140, name: 'Deniz Yıldırım', role: 'Kalite Kontrol Asistanı', email: 'deniz.yildirim@yeninsaat.com', phone: '555-101-2136', departmentId: 14, companyId: 3 },

  // Departman 15: Yazılım Geliştirme (10 kullanıcı)
  { id: 141, name: 'Ali Veli', role: 'Full Stack Developer', email: 'ali.veli@alphayazilim.com', phone: '555-112-3349', departmentId: 15, companyId: 6 },
  { id: 142, name: 'Fatma Aydın', role: 'Frontend Developer', email: 'fatma.aydin@alphayazilim.com', phone: '555-223-4460', departmentId: 15, companyId: 6 },
  { id: 143, name: 'Hasan Şahin', role: 'Backend Developer', email: 'hasan.sahin@alphayazilim.com', phone: '555-334-5571', departmentId: 15, companyId: 6 },
  { id: 144, name: 'Selin Yıldız', role: 'Mobile Developer', email: 'selin.yildiz@alphayazilim.com', phone: '555-445-6682', departmentId: 15, companyId: 6 },
  { id: 145, name: 'Burak Sarı', role: 'DevOps Engineer', email: 'burak.sari@alphayazilim.com', phone: '555-556-7793', departmentId: 15, companyId: 6 },
  { id: 146, name: 'Derya Toprak', role: 'QA Tester', email: 'derya.toprak@alphayazilim.com', phone: '555-667-8894', departmentId: 15, companyId: 6 },
  { id: 147, name: 'Emre Korkmaz', role: 'UI/UX Designer', email: 'emre.korkmaz@alphayazilim.com', phone: '555-778-9905', departmentId: 15, companyId: 6 },
  { id: 148, name: 'Gizem Yıldırım', role: 'Data Scientist', email: 'gizem.yildirim@alphayazilim.com', phone: '555-889-0016', departmentId: 15, companyId: 6 },
  { id: 149, name: 'İbrahim Efe', role: 'Software Architect', email: 'ibrahim.efe@alphayazilim.com', phone: '555-990-1127', departmentId: 15, companyId: 6 },
  { id: 150, name: 'Jale Güneş', role: 'Project Manager', email: 'jale.gunes@alphayazilim.com', phone: '555-101-2137', departmentId: 15, companyId: 6 },
];
